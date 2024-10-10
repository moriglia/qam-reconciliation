from libc.stdlib cimport malloc, free
cimport cython
cimport numpy as np
import numpy as np
#from cython.view cimport array as cvarray
from libc.math cimport exp as cexp, log as cln, abs as cabs
from cython.parallel cimport prange


cdef struct decoderResult:
    unsigned char   success
    int             iterations
    double        * final_lappr
    long            final_lappr_size


cdef inline int __sgn(double x) nogil:
    return (0.0 < x) - (x < 0.0)


cdef inline double __box_plus(double a, double b) nogil:
    return __sgn(a) * __sgn(b)\
        * min(cabs(a), cabs(b)) \
        + cln(1+cexp(-cabs(a + b))) \
        - cln(1+cexp(-cabs(a - b)))


cdef void __free_table(long** table, long tsize) noexcept nogil:
    cdef int i;
    if (table):
        for i in range(tsize):
            if (table[i]):
                free(table[i])
            else:
                break
        free(table)
    return


cdef long** __build_table(long[:] e_to_x, int num_x) noexcept:
    cdef long i, e, l;
    cdef long** table;
    cdef list edges_i;

    table = <long**> malloc(sizeof(long*)*num_x);
    if not table:
        return <long**>0
    
    for i in range(num_x):
        # look for edges involving node i 
        edges_i = [];
        l = 0;
        for e in range(e_to_x.size):
            if (e_to_x[e]==i):
                edges_i.append(e)
                l += 1

        # store the edges in the table
        # using the first element of the table
        # entry as the size of the entry
        table[i] = <long*>malloc(sizeof(long)*(l+1))
        if not table[i]:
            __free_table(table, i)
            return <long**>0
        table[i][0] = l;
        for e in range(len(edges_i)):
            table[i][e+1] = edges_i[e];
    
    return table;


cdef class Decoder:
    def __cinit__(self, long [:] e_to_v, long [:] e_to_c):
        cdef long i, c

        if (e_to_v.size != e_to_c.size):
            raise ValueError("Sizes don't match")
        
        self.__e_to_c = e_to_c
        self.__e_to_v = e_to_v
        self.__edge_num = e_to_c.size
        self.__chk_num  = max(e_to_c) + 1
        self.__var_num  = max(e_to_v) + 1

        # Reverse tables
        self.__v_to_e = __build_table(e_to_v, self.__var_num)
        if not self.__v_to_e:
            raise MemoryError()

        self.__c_to_e = __build_table(e_to_c, self.__chk_num)
        if not self.__c_to_e:
            __free_table(self.__v_to_e, self.__var_num)
            raise MemoryError()

        self.__c_to_v = <long**>malloc(sizeof(long*)*self.__chk_num)
        if not self.__c_to_v:
            __free_table(self.__v_to_e, self.__var_num)
            __free_table(self.__c_to_e, self.__chk_num)
            raise MemoryError()
        for c in range(self.__chk_num):
            self.__c_to_v[c] = <long*>malloc(sizeof(long)*(self.__c_to_e[c][0]+1))
            if not self.__c_to_v[c]:
                __free_table(self.__c_to_v, c)
                __free_table(self.__v_to_e, self.__var_num)
                __free_table(self.__c_to_e, self.__chk_num)
                raise MemoryError()
            self.__c_to_v[c][0] = self.__c_to_e[c][0]
            for i in range(1, self.__c_to_e[c][0]+1):
                self.__c_to_v[c][i] = self.__e_to_v[self.__c_to_e[c][i]]

        self.__var_to_check = None
        self.__check_to_var = None
        self.__updated_lappr= NULL
        return

    
    def __dealloc__(self):
        __free_table(self.__v_to_e, self.__var_num)
        __free_table(self.__c_to_e, self.__chk_num)
        __free_table(self.__c_to_v, self.__chk_num)
        return

    
    @property
    def cnum(self):
        """ Number of check nodes """
        return self.__chk_num

    
    @property
    def vnum(self):
        """ Number of variable nodes """
        return self.__var_num


    @property
    def ednum(self):
        """ Number of edges """
        return self.__edge_num



    """ Syndrome checking functions """    
    cdef unsigned char __check_synd_node(self, long check_node_index) noexcept nogil:
        cdef long * vnode_set
        cdef int i
        cdef unsigned char parity
        with cython.boundscheck(False):
            parity = self.__synd[check_node_index]

        vnode_set = self.__c_to_v[check_node_index]
        with cython.boundscheck(False):
            for i in range(1, vnode_set[0]+1):
                # toggle parity every time the word bit is 1
                parity ^= self.__word[vnode_set[i]]
        return parity ^ 0b1


    cpdef unsigned char check_synd_node(self,
                                        long check_node_index,
                                        unsigned char [:] word,
                                        unsigned char [:] synd):
        cdef unsigned char res

        if (word.size != self.__var_num):
            raise ValueError("Size of word does not match number of vnodes")
        if (synd.size != self.__chk_num):
            raise ValueError("Size of synd does not match number of cnodes")

        self.__word = word
        self.__synd = synd

        res = self.__check_synd_node(check_node_index)

        self.__word = None
        self.__synd = None

        return res

    
    cdef unsigned char __check_word(self) nogil:
        cdef long c_index
        for c_index in range(self.__chk_num):
            if not self.__check_synd_node(c_index):
                return 0
        return 0b1

    
    cpdef unsigned char check_word(self,
                                   unsigned char [:] word,
                                   unsigned char [:] synd):
        cdef unsigned char res
        self.__word = word
        self.__synd = synd

        res = self.__check_word()

        self.__word  = None
        self.__synd  = None

        return res

            
    cdef unsigned char __check_lappr_node(self, long check_node_index) noexcept nogil:
        cdef int i
        cdef long * vnode_set = self.__c_to_v[check_node_index]
        cdef unsigned char parity
        
        with cython.boundscheck(False):
            parity = self.__synd[check_node_index]
            
            # for i in range(1, index_set[0]+1):
            for i in range(1, vnode_set[0]+1):
                if (self.__updated_lappr[vnode_set[i]] < 0):
                    # toggle parity every time a negative lappr is met
                    parity ^= 0b1
        
        return parity ^ 0b1


    cdef unsigned char __check_lappr(self) noexcept nogil:
        cdef long i

        for i in range(self.__chk_num):
            if not self.__check_lappr_node(i):
                return 0
        return 1
    
    
    cpdef unsigned char check_lappr(self,
                                    double [:] lappr,
                                    unsigned char [:] synd):
        cdef unsigned char res

        
        if (lappr.size != self.__var_num):
            raise ValueError("Size of lappr does not match number of vnodes")
        if (synd.size != self.__chk_num):
            raise ValueError("Size of synd does not match number of cnodes")
        
        
        self.__updated_lappr = &lappr[0]
        self.__synd = synd

        res = self.__check_lappr()

        self.__word = None
        self.__synd = None
        self.__updated_lappr = NULL

        return res
    

    """ Message passing processing functions """
    cdef void __process_var_node(self, long node_index) noexcept nogil:
        cdef long* index_set
        cdef int i
        index_set = self.__v_to_e[node_index]

        with cython.boundscheck(False):
            self.__updated_lappr[node_index] = self.__lappr_data[node_index]
            for i in range(1, index_set[0]+1):
                self.__updated_lappr[node_index] += self.__check_to_var[index_set[i]]
                
            for i in range(1, index_set[0]+1):
                self.__var_to_check[index_set[i]] = self.__updated_lappr[node_index] \
                    - self.__check_to_var[index_set[i]]
        return


    cpdef void process_var_node(self,
                                long node_index,
                                double [:] lappr_data,
                                double [:] check_to_var,
                                double [:] var_to_check,
                                double [:] updated_lappr):
        self.__lappr_data = lappr_data
        self.__check_to_var = check_to_var
        self.__var_to_check = var_to_check
        self.__updated_lappr = &updated_lappr[0]
        
        self.__process_var_node(node_index)

        self.__lappr_data    = None
        self.__check_to_var  = None
        self.__var_to_check  = None
        self.__updated_lappr = NULL

        return

            
    cdef int __process_check_node(self, long node_index) noexcept nogil:
        cdef long* index_set
        cdef double prefactor
        cdef int i, j
        cdef long N_vn
        cdef double _atanh, _msg
        cdef double * F
        cdef double * B
        
        index_set = self.__c_to_e[node_index]
        N_vn = index_set[0]
        index_set+=1
        F = <double*>malloc((N_vn-1)*sizeof(double)*2)
        if F is NULL:
            return 1

        with cython.boundscheck(False):
            B = F + N_vn - 2
            # Note that &B[0] == &F[N_nv - 2] (which is the last location "used by" F )
            # But B[0] is actually never used

            F[0]      = self.__var_to_check[index_set[0]]
            B[N_vn-1] = self.__var_to_check[index_set[N_vn-1]]

            j=0
            for i in range(1,N_vn-1):
                # in this loop j is always i-1
                _msg = self.__var_to_check[index_set[i]]
                F[i] = __box_plus(F[j], _msg)
                j = i

            j = N_vn-1
            for i in range(N_vn-2, 0, -1):
                # in this loop, j is always i+1
                _msg = self.__var_to_check[index_set[i]]
                B[i] = __box_plus(B[j], _msg)
                j = i
            
            prefactor = -1.0 if self.__synd[node_index] else 1.0
            self.__check_to_var[index_set[0]] = prefactor*B[1]

            for i in range(1, N_vn-1):
                self.__check_to_var[index_set[i]] = prefactor * __box_plus(F[i-1], B[i+1])

            self.__check_to_var[index_set[N_vn-1]] = prefactor*F[N_vn-2]
            
        free(F)
        return 0

    
    cpdef int process_check_node(self,
                                 long node_index,
                                 unsigned char [:] synd,
                                 double [:] check_to_var,
                                 double [:] var_to_check):
        cdef int res
        self.__check_to_var = check_to_var
        self.__var_to_check = var_to_check
        self.__synd         = synd

        res = self.__process_check_node(node_index)

        self.__check_to_var = None
        self.__var_to_check = None
        self.__synd         = None
            
        return res
    
        
    cdef decoderResult _decode(self,
                               double [:] lappr_data,
                               unsigned char [:] synd,
                               int max_iterations) nogil:

        cdef long iter_index, c, v, e
        cdef decoderResult res

        res.final_lappr_size = self.__var_num

        with cython.boundscheck(False):
            self.__updated_lappr = &lappr_data[0]
        self.__synd          = synd
        if (self.__check_lappr()):
            self.__updated_lappr = NULL
            res.success     = 1
            res.iterations  = 0
            with cython.boundscheck(False):
                res.final_lappr = &lappr_data[0]
            return res

        with gil:
            self.__check_to_var  = np.zeros(self.__edge_num, dtype=np.double)
            self.__var_to_check  = np.empty_like(self.__check_to_var)

        # DO NOT FREE HERE: it will be embedded in a numpy array in cpdef tuple decode(...)
        self.__updated_lappr = <double*>malloc(self.__edge_num*sizeof(double))
        if self.__updated_lappr is NULL:
            raise MemoryError("Out of memory: cannot allocate the updated lappr array")
        res.final_lappr = self.__updated_lappr

        self.__lappr_data    = lappr_data
        self.__synd          = synd

        
        # First half iteration to propagate lapprs to check nodes
        # The following line also initializes var_to_check
        for v in prange(self.__var_num):
            self.__process_var_node(v)

        
        for iter_index in range(max_iterations):
            for c in prange(self.__chk_num):
                self.__process_check_node(c)
                
            for v in prange(self.__var_num):
                self.__process_var_node(v)
                
            if (self.__check_lappr()):
                self.__updated_lappr = NULL
                
                res.success    = 1
                res.iterations = iter_index+1
                
                return res

        self.__updated_lappr = NULL
        
        res.success     = 0
        res.iterations  = max_iterations
        return res


    
    
    cpdef tuple decode(self,
                       double [:] lappr_data,
                       unsigned char [:] synd,
                       int max_iterations):
        cdef decoderResult res
        
        res = self._decode(lappr_data,
                           synd,
                           max_iterations)
        # Unpack result to tuple
        return (res.success,
                res.iterations,
                np.array(<double[:res.final_lappr_size]>res.final_lappr))
