from cython.view cimport array as cvarray
from .noisemapper cimport NoiseMapper
from .alphabet cimport PAMAlphabet
from libc.math cimport sqrt, exp, log2
cimport numpy as np
import numpy as np
from scipy.integrate import quad, fixed_quad


cdef double __sqrt2 = sqrt(2)
cdef double __sqrt2pi = sqrt(2*np.pi)


cpdef double [:] P_xhat(NoiseMapper nm):
    cdef long i, j
    cdef double [:] res = cvarray(shape=(nm.order,),
                                  itemsize=sizeof(double),
                                  format="d")
    for i in range(nm.order):
        res[i] = 0
        for j in range(nm.order):
            res[i] += nm.probabilities[j] * nm.fwrd_transition_probability[j, i]

    return res



cpdef double mutual_information_base_scheme_arg(double n,
                                                NoiseMapper nm,
                                                double [:] p_Xhat):
    """
    n  : information shared by Bob to Alice
    nm : noise mapper object, with all necessary data, including noise data


    In the following, i and j will refer to:
    a_i: symbol received by Bob
    a_j: symbol transmitted by Alice
    """
    cdef long i, j, k
    cdef double y_hat, __twoVariance
    cdef double res = 0
    cdef double [:]  derivative_g = cvarray(shape=(nm.order,),
                                            itemsize=sizeof(double),
                                            format="d")
    # this ^^^^^^^^ will contain g'_i(g_i^{-1}(n)) for each i
    cdef double [:,:] f_N_Xhat_cond_X = cvarray(shape=(nm.order,nm.order),
                                          itemsize=sizeof(double),
                                          format="d")
    cdef double [:] f_N_cond_X = cvarray(shape=(nm.order,),
                                         itemsize=sizeof(double),
                                         format="d")

    __twoVariance = 2.0*nm.noise_var
    
    for i in range(nm.order):
        y_hat = nm.g_inv(n, i)


        for j in range(nm.order):
            f_N_Xhat_cond_X[i,j] = nm.probabilities[j]

            for k in range(j):
                f_N_Xhat_cond_X[i,j] += nm.probabilities[k] * \
                    exp(-(2.0*y_hat - nm.constellation[j] - nm.constellation[k]) * \
                        (nm.constellation[j] - nm.constellation[k])/__twoVariance)
            for k in range(j+1, nm.order):
                f_N_Xhat_cond_X[i,j] += nm.probabilities[k] * \
                    exp(-(2.0*y_hat - nm.constellation[j] - nm.constellation[k]) * \
                        (nm.constellation[j] - nm.constellation[k])/__twoVariance)

            f_N_Xhat_cond_X[i,j] = nm.delta_F_Y[i] / f_N_Xhat_cond_X[i,j]
        
    for j in range(nm.order):
        f_N_cond_X[j] = 0
        for i in range(nm.order):
            f_N_cond_X[j] += f_N_Xhat_cond_X[i, j]

    if np.isnan(res):
        print("Should never happen")
        return res
    

    for j in range(nm.order):
        for i in range(nm.order):
            # we use y_hat as a temporary variable, because we can save time
            # by skipping a multiplication and we do not need y_hat anymore
            y_hat = f_N_Xhat_cond_X[i, j] * nm.probabilities[j]
            if y_hat > 0.0:
                res += y_hat * log2(y_hat/p_Xhat[i])
                if np.isnan(res):
                    print("NaN in loop f_N_Xhat_cond_X")
                    print(y_hat)
                    return np.nan
        y_hat = nm.probabilities[j] * f_N_cond_X[j]
        if y_hat > 0.0:
            res -= y_hat * log2(y_hat)
            if np.isnan(res):
                print("NaN in loop f_N_cond_X")
                print(y_hat)
                return np.nan
        

    return res



cpdef double mutual_information_base_scheme(NoiseMapper nm, double [:] p_Xhat):
    # cdef NoiseMapper nm = NoiseMapper(pa, noise_var)
    # cdef double [:] p_Xhat = P_xhat(nm)
    cdef double I
    # cdef double res = 0
    # cdef long k
    # cdef double f, f_prec, n
    # cdef unsigned char nan_detected = 0

    # res = mutual_information_base_scheme_arg(0, nm, p_Xhat) / 2

    # for k in range(N):
    #     n = k/N
    #     #f = mutual_information_base_scheme_arg(n, nm, p_Xhat)
    #     res += mutual_information_base_scheme_arg(n, nm, p_Xhat) # trapezoid
    #     #f_prec = f
    #     # if np.isnan(f) and not nan_detected:
    #     #     print(f"NaN in integration at noise_var={noise_var}")
    #     #     nan_detected = 1

    # res += mutual_information_base_scheme_arg(0, nm, p_Xhat) / 2
    # res /= N

    I, _ = quad(mutual_information_base_scheme_arg, 0, 1,
                     args=(nm, p_Xhat))
    return I
    # return res


cpdef double mutual_information_X_Xhat(NoiseMapper nm, double [:] p_Xhat):
    # cdef double [:] p_Xhat = P_xhat(nm)
    cdef double res = 0.0
    cdef double sum_i, tmp
    cdef long i, j


    for j in range(nm.order):
        sum_i = 0.0
        for i in range(nm.order):
            tmp = 0.0
            if (nm.fwrd_transition_probability[j, i] > 0.0):
                tmp += log2(nm.fwrd_transition_probability[j, i])
            if (p_Xhat[i] > 0.0):
                tmp -= log2(p_Xhat[i])

            sum_i += tmp*nm.fwrd_transition_probability[j,i]

        res += nm.probabilities[j] * sum_i

    return res


cpdef double mutual_information_X_Y_int_arg(double y, NoiseMapper nm):
    cdef double res = 0
    cdef double tmp, tmp2, __twoVariance
    cdef long j, k

    __twoVariance = 2.0*nm.noise_var

    for j in range(nm.order):
        tmp = nm.probabilities[j]
        for k in range(j):
            tmp += nm.probabilities[k] * \
                exp((2*y - nm.constellation[k] - nm.constellation[j]) * \
                    (nm.constellation[k] - nm.constellation[j])/__twoVariance)
                
        for k in range(j+1, nm.order):
            tmp += nm.probabilities[k] * \
                exp((2*y - nm.constellation[k] - nm.constellation[j]) * \
                    (nm.constellation[k] - nm.constellation[j])/__twoVariance)
        tmp2 = nm.probabilities[j] * \
            exp(-(y - nm.constellation[j])*(y - nm.constellation[j])/__twoVariance) * log2(tmp)
        if not np.isnan(tmp2):
            res -= tmp2
        
    res /= sqrt(2.0*np.pi)*nm.noise_sigma
    return res


cpdef double mutual_information_X_Y(NoiseMapper nm):
    cdef double I

    I, _ = quad(mutual_information_X_Y_int_arg, -np.inf, np.inf,
                args=(nm,))

    return I



cpdef (double, double, double) montecarlo_information(
    PAMAlphabet      pa,
    NoiseMapper      nm,
    double      [:]  p_Xhat,
    long             N,
    unsigned char[:] which = np.ones(3, dtype=np.uint8)
):
    cdef long [:] x_ind = cvarray(shape=(N,),
                                  itemsize=sizeof(long),
                                  format="l")
    cdef double [:] y = cvarray(shape=(N,),
                                itemsize=sizeof(double),
                                format="d")
    cdef long [:] xhat_ind = cvarray(shape=(N,),
                                     itemsize=sizeof(long),
                                     format="l")
    cdef double [:] n = cvarray(shape=(N,),
                                itemsize=sizeof(double),
                                format="d")
    cdef long p, k, m, curr_x_ind, curr_xhat_ind
    cdef double I_X_Xhat, I_X_Y, I_XN_Xhat,
    cdef double tmp, __twoVariance, y_hat, I_XN_Xhat_tmp
    cdef double x, xhat
    
    x_ind[:] = pa.random_symbols(N)
    y[:] = pa.index_to_value(x_ind)
    for p in range(N):
        y[p] += nm.noise_sigma*np.random.randn()

    xhat_ind = nm.hard_decide_index(y)
    n = nm.map_noise(y, xhat_ind) # here n is exactly what it is ment to be


    I_X_Xhat = 0
    I_X_Y = 0
    I_XN_Xhat = 0

    __twoVariance = 2.0 * nm.noise_var
    
    for p in range(N):
        curr_x_ind    = x_ind[p]
        curr_xhat_ind = xhat_ind[p]
        x             = pa.constellation[curr_x_ind]


        # I(X ; Xhat) ----------------------------------------------------
        if (which[0]):
            I_X_Xhat += log2(p_Xhat[curr_xhat_ind]/nm.fwrd_transition_probability[curr_x_ind, curr_xhat_ind])


        # I(X ; Y) -------------------------------------------------------
        if (which[1]):
            tmp = pa.probabilities[curr_x_ind]
            for k in (*range(curr_x_ind), *range(curr_x_ind+1, pa.order)):
                tmp += pa.probabilities[k] * \
                    exp((2*y[p]-pa.constellation[k]-x)*(pa.constellation[k]-x)/__twoVariance)
                
            I_X_Y += log2(tmp)


        # I(X,N ; Xhat) --------------------------------------------------
        if (which[2]):
            I_XN_Xhat_tmp = 0
            for k in (*range(curr_xhat_ind), *range(curr_xhat_ind+1, pa.order)):
                y_hat = nm.g_inv(n[p], k)
                tmp = pa.probabilities[curr_x_ind]
                for m in (*range(curr_x_ind), *range(curr_x_ind+1, pa.order)):
                    tmp += pa.probabilities[m] * \
                        exp((2*y_hat - x - pa.constellation[m])*(pa.constellation[m]-x)/__twoVariance)
                I_XN_Xhat_tmp += nm.delta_F_Y[k] / tmp

            tmp = pa.probabilities[curr_x_ind]
            y_hat = nm.g_inv_search(n[p], curr_xhat_ind)
            for m in (*range(curr_x_ind), *range(curr_x_ind+1, pa.order)):
                tmp += pa.probabilities[m] * \
                    exp((2*y_hat - x - pa.constellation[m])*(pa.constellation[m]-x)/__twoVariance)
            I_XN_Xhat_tmp *= tmp / nm.delta_F_Y[curr_xhat_ind]
            I_XN_Xhat_tmp += 1
            I_XN_Xhat_tmp *= p_Xhat[curr_xhat_ind]
            
            I_XN_Xhat -= log2(I_XN_Xhat_tmp)


    I_X_Xhat  /= N
    I_X_Y     /= N
    I_XN_Xhat /= N


    return I_X_Xhat, I_X_Y, I_XN_Xhat
