from .alphabet cimport PAMAlphabet, Alphabet
import scipy as sp
cimport numpy as np
import numpy as np
from cython.view cimport array as cvarray
from libc.math cimport sqrt, erf, exp, log


cdef double __sqrt2 = sqrt(2)


cdef long __binsearch(double [:] domain, double val):
    cdef long index

    if (domain.size==1):
        return 0
    if (val < domain[0]):
        return 0
    if (val > domain[-1]):
        return domain.size-1

    index = domain.size//2-1
    
    if (val < domain[index]):
        return __binsearch(domain[:index], val)
    if (val >= domain[index+1]):
        return index + 1 + __binsearch(domain[index+1:], val)

    return index


cdef double __interp(double [:] domain, double [:] codomain, double val):
    cdef long index

    if (val >= domain[-1]):
        return codomain[-1]
    
    index = __binsearch(domain, val)

    if (index == domain.size - 1):
        return codomain[index]

    if (domain[index+1]==domain[index]):
        return codomain[index]
    
    return codomain[index] + \
        (codomain[index+1] - codomain[index]) * \
        (val - domain[index])/(domain[index+1] - domain[index])


cdef double __F_Z(double z, double mu, double sigma):
    return 0.5*(1+sp.special.erf((z-mu)/(__sqrt2*sigma)))


cpdef double [:] F_Z(double [:] z, double mu, double sigma):
    cdef double [:] res = cvarray(shape=(z.size,),
                                  itemsize=sizeof(double),
                                  format="d")

    cdef int i
    for i in range(z.size):
        res[i] = __F_Z(z[i], mu, sigma)

    return res


cdef double __dist_cut(double x):
    if (x<0):
        return 0
    if (x>=1):
        return 1
    return x


cpdef double[:] __view_dist_cut(double [:] x):
    cdef int i
    cdef double [:] res = cvarray(shape=(x.size,),
                                  itemsize=sizeof(double),
                                  format="d")
    for i in range(x.size):
        res[i] = __dist_cut(x[i])

    return res



cdef class NoiseMapper:
    def __cinit__(self, PAMAlphabet pa, double noise_var):
        cdef int i, j, k
        cdef double tmp
        
        if (noise_var <= 0):
            raise ValueError(f"noise variance must be strictly positive, got {noise_var}")
        
        # Alphabet internals
        self.order = pa.order
        self.half_order = pa.order >> 1
        self.bit_per_symbol = pa.bit_per_symbol
        self.constellation = pa.constellation
        self.variance = pa.variance
        self.thresholds = pa.thresholds
        self.probabilities = pa.probabilities
        
        self.noise_var = noise_var
        self.__sigma = sqrt(noise_var)

        self.__y_low = pa.constellation[0]*3
        self.__y_high= pa.constellation[-1]*3
        self.__n_points = 10000      
        self._y_range = np.linspace(self.__y_low, self.__y_high, self.__n_points)
        self._F_Y = self.F_Y(self._y_range)
        
        self.__ref_symb = pa.order//2
        self.__ref_symb_value = pa.constellation[self.__ref_symb]

        self.F_Y_thresholds = np.empty(pa.order+1, dtype=np.double)
        self.F_Y_thresholds[0] = 0
        self.F_Y_thresholds[pa.order] = 1
        for i in range(1, pa.order):
            self.F_Y_thresholds[i] = self._single_F_Y(pa.thresholds[i])
        # self.__ref_F_Y_threshold = self.F_Y_thresholds[self.__ref_symb]
        
        self.delta_F_Y = cvarray(shape=(pa.order,),
                                 itemsize=sizeof(double),
                                 format="d")
        for i in range(pa.order):
            self.delta_F_Y[i] = \
                self.F_Y_thresholds[i+1] - \
                self.F_Y_thresholds[i]

        # self.__ref_delta_F_Y = self.__delta_F_Y[self.__ref_symb]

        # Crate table with probabilities P{ X_hat=a_i | X=a_j }
        self.fwrd_transition_probability = cvarray(shape=(pa.order, pa.order),
                                                   itemsize=sizeof(double),
                                                   format="d")
        tmp = __sqrt2*self.__sigma
        for j in range(pa.order): # Index of transmitted symbol
            # i == 0
            self.fwrd_transition_probability[j, 0] = \
                0.5*(erf((self.thresholds[1]-self.constellation[j])/tmp) + 1)
            # i == pa.order-1
            self.fwrd_transition_probability[j, pa.order-1] = \
                0.5*(1 - erf((self.thresholds[pa.order-1]-self.constellation[j])/tmp))
            # All standard cases
            for i in range(1, pa.order-1):
                self.fwrd_transition_probability[j,i] = \
                    0.5*(erf((self.thresholds[i+1]-self.constellation[j])/tmp) - \
                         erf((self.thresholds[i]-self.constellation[j])/tmp))

        # Crate table with probabilities P{ x=constellation[j] | x_hat=constellation[i] }
        self.back_transition_probability = cvarray(shape=(pa.order, pa.order),
                                                   itemsize=sizeof(double),
                                                   format="d")
        for i in range(pa.order):
            for j in range(pa.order):
                tmp = 0
                for k in range(pa.order):
                    tmp += self.probabilities[k] * self.fwrd_transition_probability[k, i]
                self.back_transition_probability[i,j] = \
                    self.probabilities[j] * self.fwrd_transition_probability[j,i] / tmp


        # Create bare LLR table 
        self.bare_llr_table = cvarray(shape=(pa.order, self.bit_per_symbol),
                                      itemsize=sizeof(double),
                                      format="d")
        cdef double __N, __D
        cdef int __mode_index
        for j in range(pa.order): # index of transmitted symbol
            for k in range(self.bit_per_symbol): # bit index
                __N = 0
                __D = 0
                for i in range(pa.order): # index of symbol received by Bob
                    __mod_index = i>>k
                    
                    if ((__mod_index*(__mod_index +1)) & 0b11):
                        # __mod_index is congruent either to 1 or to 2 (mod 4)
                        __D += self.fwrd_transition_probability[j, i]
                    else:
                        # __mod_index is congruent either to 0 or to 3 (mod 4)
                        __N += self.fwrd_transition_probability[j, i]
                # Finally for symbol a_j, the LLR of bit b_k is:
                self.bare_llr_table[j,k] = log(__N / __D)


        self.inf_erf_table = cvarray(shape=(pa.order, pa.order),
                                     itemsize=sizeof(double),
                                     format="d")

        
        # inf_erf_table[i,j] = ERF((inf(D_i) - a_j)/(2\sigma^2))
        tmp = __sqrt2*self.__sigma
        for j in range(pa.order):
            self.inf_erf_table[0, j] = -1
            for i in range(1, pa.order):
                self.inf_erf_table[i,j] = erf(
                    (pa.thresholds[i]-pa.constellation[j])/tmp
                )
        return


    # @property
    # def ref_symb(self):
    #     return self.__ref_symb


    # @property
    # def ref_symb_value(self):
    #     return self.__ref_symb_value


    # @property
    # def delta_F_Y(self):
    #     return np.array(self.__delta_F_Y)


    @property
    def y_range(self):
        return np.array(self._y_range)

    
    @property
    def F_Y_values(self):
        return np.array(self._F_Y)


    cpdef double [:] F_Y(self, double [:] y):
        cdef int i, j
        cdef double [:] res
        # res[:] = cvarray(size=(y.size,),
        #               itemsize=sizeof(double),
        #               format="i")
        res = F_Z(y, self.constellation[0], self.__sigma)
        for j in range(y.size):
            for i in range(1, self.order):
                res[j] += __F_Z(y[j], self.constellation[i], self.__sigma)
            res[j] /= self.order
        return res
    
    
    cdef double _single_F_Y(self, double y):
        cdef int i
        cdef double res

        res = __F_Z(y, self.constellation[0], self.__sigma) * self.probabilities[0]
        for i in range(1, self.order):
            res += __F_Z(y, self.constellation[i], self.__sigma) * self.probabilities[i]
        
        return res


    cdef double g(self, double y, int i):
        return (self._single_F_Y(y) - self.F_Y_thresholds[i])/self.delta_F_Y[i]

    
    cdef double g_inv(self, double n_hat, int i):
        """ Note that this returns y_hat and not z_hat """
        return __interp(
            self._F_Y,
            self._y_range,
            n_hat * self.delta_F_Y[i] + self.F_Y_thresholds[i]
        )


    cpdef long [:] hard_decide_index (self, double [:] y_samples):
        cdef long j
        cdef long [:] res = cvarray(shape=(y_samples.size,),
                                    itemsize=sizeof(long),
                                    format="l")
        for j in range(y_samples.size):
            res[j] = __binsearch(self.thresholds,
                                 y_samples[j])
            if (res[j]==self.order):
                res[j] = self.order-1
        return res


    cpdef double [:] index_to_val (self, long [:] index):
        cdef int i
        cdef double [:] res = cvarray(shape=(index.size,),
                                      itemsize=sizeof(double),
                                      format="d")
        for i in range(index.size):
            res[i] = self.constellation[index[i]]

        return res

        
    cpdef double [:] map_noise(self, double [:] y_samples, long[:] index):
        # cdef long i, j
        cdef long j
        cdef double [:] res

        if y_samples.size != index.size:
            raise ValueError(f"Input vectors sizes do not match")
        
        res = cvarray(shape=(y_samples.size,),
                      itemsize=sizeof(double),
                      format="d")

        for j in range(y_samples.size):
            res[j] = self.g(y_samples[j], index[j])

        return res


    cpdef double [:] demap_noise(self, double [:] n_hat, long [:] symb):
        cdef double [:] y_hat
        cdef int i
        if (n_hat.size != symb.size):
            raise ValueError("Sizes do not match")

        y_hat = cvarray(shape=(n_hat.size,),
                        itemsize=sizeof(double),
                        format="d")

        for i in range(n_hat.size):
            y_hat[i] = self.__g_inv(n_hat[i], symb[i])
        return y_hat



    cpdef double [:] bare_llr(self, long [:] symb):
        cdef int i
        cdef double [:] llr = cvarray(shape=(self.bit_per_symbol*symb.size,),
                                      itemsize=sizeof(double),
                                      format="d")

        for i in range(symb.size):
            llr[i*self.bit_per_symbol : (i+1)*self.bit_per_symbol] = self.bare_llr_table[symb[i],:]

        return llr
    
    
    
    
    
# cdef class NoiseDemapper(NoiseMapper):
#     pass
#     # Use the same __cinit__: note that, as specified
#     # in the Cython documentation, __cinit__ is automatically called
#     # and its argument list cannot be modified
#     # def __cinit__(self, PAMAlphabet pa, double noise_var):
#     #     cdef long i, j
#     #     cdef double __tmp
        
#     #     return
        
    """ Formulation 2/4 in notes """
    cpdef double [:] demap_lappr(self, double n, long j):
        """ procedure to construct the LAPPRs of the bits
        associated to a transmitted symbol
        
        Inputs:
        - double n: transformed noise, given by Bob
        - long   j: index of transmitted symbol: a_j
        
        Output:
        - double [log2 M] lapprs:
            log A Posteriori Probability ratios of the log2(M) bits
            associated to the remote symbol
        
        Suppose that Alice (who is running this function), has transmitted a_j.
        Bob has sent her n
        """
        cdef long i, k
        cdef double y_hat_i
        cdef double sums_i
        cdef double __2sigmasquare = 2*self.noise_var

        cdef double [:] N     = cvarray(shape=(self.bit_per_symbol,),
                                        itemsize=sizeof(double),
                                        format="d")
        cdef double [:] D     = cvarray(shape=(self.bit_per_symbol,),
                                        itemsize=sizeof(double),
                                        format="d")
        cdef double [:] lappr = cvarray(shape=(self.bit_per_symbol,),
                                        itemsize=sizeof(double),
                                        format="d")

        cdef int mod_index;


        # Initialization of numerators and denominators of lapprs:
        for k in range(self.bit_per_symbol):
            N[k] = 0;
            D[k] = 0;
        

        for i in range(self.order):
            # Part I:
            # For each possible symbol $a_i$ decided by Bob
            # construct the sample from which Bob should
            # have decided for that symbol, based on what
            # Alice knows about that sample: its transformed
            # noise estimation is equal to $n$
            y_hat_i = self.g_inv(n, i)


            # Part II
            # Sum of the probability weighted exponentials
            sums_i = 0;
            for k in range(j):
                sums_i += exp(
                    (2*y_hat_i-self.constellation[k]-self.constellation[j]) * \
                    (self.constellation[k] - self.constellation[j])
                ) * self.probabilities[k]

            sums_i += self.probabilities[j]

            for k in range(j+1, self.order):
                sums_i += exp(
                    (2*y_hat_i-self.constellation[k]-self.constellation[j]) * \
                    (self.constellation[k] - self.constellation[j])/__2sigmasquare
                ) * self.probabilities[k]


            # Part III
            # Add the newly evaluated conditional probability
            # to the numerator or denominator
            mod_index = i;
            for k in range(self.bit_per_symbol):
                if ((mod_index * (mod_index+1)) & 0b11):
                    # mod_index is either 1 or 2
                    D[k] += self.delta_F_Y[i] / sums_i
                else:
                    # mod_index is either 0 or 3
                    N[k] += self.delta_F_Y[i] / sums_i
                    
                mod_index >>= 1
            
            


        # Finalization
        # Eval LAPPRs
        for k in range(self.bit_per_symbol):
            lappr[k] = log(N[k]) - log(D[k])
            
        return lappr



    cpdef double [:] demap_lappr_array(self, double [:] n, long [:] j):
        cdef double [:] lappr
        cdef int i
        
        if (n.size != j.size):
            raise ValueError("Sizes of transformed noise vector and tx symbols do not match")

        lappr = cvarray(shape=(n.size*self.bit_per_symbol,),
                        itemsize=sizeof(double),
                        format="d")

        for i in range(n.size):
            lappr[i*self.bit_per_symbol : (i+1)*self.bit_per_symbol] = \
                self.demap_lappr(n[i], j[i])

        return lappr


    """ Formulation 1 in my notes """
    cpdef double [:] demap_lappr_simplified(self, double n, long j):
        cdef long k, i, __mod_index
        cdef double y_hat_i
        cdef double a_i, a_j
        cdef double __2sigmasquare
        
        cdef double [:] N     = cvarray(shape=(self.bit_per_symbol,),
                                        itemsize=sizeof(double),
                                        format="d")
        cdef double [:] D     = cvarray(shape=(self.bit_per_symbol,),
                                        itemsize=sizeof(double),
                                        format="d")
        cdef double [:] lappr = cvarray(shape=(self.bit_per_symbol,),
                                        itemsize=sizeof(double),
                                        format="d")

        a_j = self.constellation[j]
        __2sigmasquare = 2*self.noise_var
        
        
        for k in range(self.bit_per_symbol):
            N[k] = 0;
            D[k] = 0;

        for i in range(self.order):
            y_hat_i = self.g_inv(n, i)
            __mod_index = i
            for k in range(self.bit_per_symbol):
                if ((__mod_index*(__mod_index+1)) & 0b11):
                    D[k] += exp(-((y_hat_i - a_j)**2)/__2sigmasquare)
                else:
                    N[k] += exp(-((y_hat_i - a_j)**2)/__2sigmasquare)

                __mod_index >>= 1

        for k in range(self.bit_per_symbol):
            lappr[k] = log(N[k]) - log(D[k])

        return lappr


            
    cpdef double [:] demap_lappr_simplified_array(self, double [:] n, long [:] j):
        cdef double [:] lappr
        cdef int i
        
        if (n.size != j.size):
            raise ValueError("Sizes of transformed noise vector and tx symbols do not match")

        lappr = cvarray(shape=(n.size*self.bit_per_symbol,),
                        itemsize=sizeof(double),
                        format="d")

        for i in range(n.size):
            lappr[i*self.bit_per_symbol : (i+1)*self.bit_per_symbol] = \
                self.demap_lappr_simplified(n[i], j[i])

        return lappr


    """ Formulation 3 in my notes """
    cpdef double [:] demap_lappr_sofisticated(self, double n, long j):
        cdef long i, m, l, __mod_index
        cdef double [:] y_hat = cvarray(shape=(self.order,),
                                        itemsize=sizeof(double),
                                        format="d")
        cdef double e_coeff, __2sigmasquare, __sqrt2sigma, a_j, A
        cdef double S_zj, B_j
        cdef double [:] beta = cvarray(shape=(self.order,),
                                       itemsize=sizeof(double),
                                       format="d")
        cdef double [:] delta_F_Z = cvarray(shape=(self.order,),
                                            itemsize=sizeof(double),
                                            format="d")
        # cdef double [:] delta_bottom_up = cvarray(shape=(self.order-1,),
        #                                           itemsize=sizeof(double),
        #                                           format="d")
        # cdef double [:] delta_top_down = cvarray(shape=(self.order-1,),
        #                                          itemsize=sizeof(double),
        #                                          format="d")
        cdef double [:] lappr = cvarray(shape=(self.bit_per_symbol,),
                                        itemsize=sizeof(double),
                                        format="d")
        cdef double [:] N = cvarray(shape=(self.bit_per_symbol,),
                                    itemsize=sizeof(double),
                                    format="d")
        cdef double [:] D = cvarray(shape=(self.bit_per_symbol,),
                                    itemsize=sizeof(double),
                                    format="d")
        
        """ Part 1: Hypotetical samples re-construction """
        for i in range(self.order):
            y_hat[i] = self.g_inv(n, j) # Hypotetical sample

        """ Part 2: Coefficient list re-construction """
        __2sigmasquare = 2*self.noise_var
        __sqrt2sigma = sqrt(__2sigmasquare)
        a_j = self.constellation[j]
        S_zj = 0
        B_j = 0
        for i in range(self.order):
            """ Part 2A: Exponential coefficients weighted sum """
            e_coeff = self.probabilities[j]
            for m in range(j):
                e_coeff += self.probabilities[m] * exp(
                    (2*y_hat[i] - self.constellation[m]-a_j) * \
                    (self.constellation[m]-a_j) / __2sigmasquare
                )
            for m in range(j+1, self.order):
                e_coeff += self.probabilities[m] * exp(
                    (2*y_hat[i] - self.constellation[m]-self.constellation[j]) * \
                    (self.constellation[m] - a_j) / __2sigmasquare
                )
                
            """ Part 2B: beta coefficients calculation """
            beta[i] = self.delta_F_Y[i] / e_coeff
            B_j += beta[i]
            
            """ Part 2C: delta F_Z coefficients """
            delta_F_Z[i] = 0.5*(erf((y_hat[i] - a_j) / __sqrt2sigma) - self.inf_erf_table[i,j])
            S_zj += delta_F_Z[i]

        # """ Part 3: Auxiliary sum of Cryptic difference coefficients:
        # Part 3A: bottom differences partial sum
        # Part 3B: top differences partial sum
        # """
        # # Part 3A: Note that delta_bottom_up is defined for i=0, ..., M-2
        # delta_bottom_up[0] = delta_F_Z[0] - beta[0]
        # for i in range(1, self.order - 1):
        #     delta_bottom_up[i] = delta_F_Z[i] - beta[i] + delta_bottom_up[i-1]

        # """ Part 3B:
        # Note that delta_top_down is formally defined for i=1, ..., M-1,
        # but for the sake of the algorithm we re-map it to the range: i=0,...,M-2 """
        # delta_top_down[self.order-2] = delta_F_Z[self.order-2] - beta[self.order-2]
        # for i in range(self.order-3, -1, -1):
        #     delta_top_down[i] = delta_top_down[i+1] + delta_F_Z[i] - beta[i]

        """ Part 4: LAPPR construction """
        # for l in range(self.bit_per_symbol-1):
        #     # N[l] is initialized by the the edge case i == 0
        #     D[l] = 0
        #     # D[log2(M)-1] is initialized by the edge case i == M-1

        # # Edge case i == 0
        # A = beta[0]*delta_F_Z[0]*delta_top_down[0]
        # for l in range(self.bit_per_symbol):
        #     N[l] = A

        # # Edge case i == M-1
        # A = beta[self.order-1] * delta_F_Z[self.order-1] * delta_bottom_up[self.order-2]
        # for l in range(self.bit_per_symbol-1):
        #     N[l] += A
        # D[self.bit_per_symbol-1] = A

        # # Ordinary cases
        # for i in range(1, self.order-1):
        #     A = beta[i]*delta_F_Z[i]*(delta_bottom_up[i-1] + delta_top_down[i])
        #     __mod_index = i
        #     for l in range(self.bit_per_symbol):
        #         if ((__mod_index*(__mod_index+1)) & 0b11):
        #             D[l] += A
        #         else:
        #             N[l] += A
        #         __mod_index >>= 1

        for l in range(self.bit_per_symbol):
            D[l] = 0
            N[l] = 0
            
        for i in range(self.order):
            A = beta[i]*S_zj - delta_F_Z[i]*B_j
            __mod_index = i
            for l in range(self.bit_per_symbol):
                if ((__mod_index*(__mod_index+1)) & 0b11):
                    D[l] += A
                else:
                    N[l] += A
                __mod_index >>= 1

        # LAPPRs
        for l in range(self.bit_per_symbol):
            lappr[l] = log(N[l]) - log(D[l])

        return lappr

        

    cpdef double [:] demap_lappr_sofisticated_array(self, double [:] n, long [:] j):
        cdef double [:] lappr
        cdef int i
        
        if (n.size != j.size):
            raise ValueError("Sizes of transformed noise vector and tx symbols do not match")

        lappr = cvarray(shape=(n.size*self.bit_per_symbol,),
                        itemsize=sizeof(double),
                        format="d")

        for i in range(n.size):
            lappr[i*self.bit_per_symbol : (i+1)*self.bit_per_symbol] = \
                self.demap_lappr_sofisticated(n[i], j[i])

        return lappr








cdef class NoiseMapperFlipSign(NoiseMapper):
    cdef double g(self, double y, int i):
        if (i<self.half_order):
            # return self.g(-y, self.order - 1 - i)
            return (self.F_Y_thresholds[i+1] - self._single_F_Y(y))/self.delta_F_Y[i]
        return (self._single_F_Y(y) - self.F_Y_thresholds[i])/self.delta_F_Y[i]

    
    cdef double g_inv(self, double n_hat, int i):
        """ Note that this returns y_hat and not z_hat """
        if (i<self.half_order):
            return __interp(
                self._F_Y,
                self._y_range,
                self.F_Y_thresholds[i+1] - n_hat * self.delta_F_Y[i]
            )
        return __interp(
            self._F_Y,
            self._y_range,
            n_hat * self.delta_F_Y[i] + self.F_Y_thresholds[i]
        )
