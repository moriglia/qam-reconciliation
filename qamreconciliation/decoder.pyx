from libc.stdlib cimport malloc, free
cimport cython
cimport numpy as np
import numpy as np
#from cython.view cimport array as cvarray
from libc.math cimport exp as cexp, log as cln, abs as cabs


cdef inline int __sgn(double x) nogil:
    return (0.0 < x) - (x < 0.0)


# cdef inline void __free_array(void* arr):
#     if arr:
#         free(arr)
#     return


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
        self.__updated_lappr= None
        return

    
    def __dealloc__(self):
        __free_table(self.__v_to_e, self.__var_num)
        __free_table(self.__c_to_e, self.__chk_num)
        __free_table(self.__c_to_v, self.__chk_num)
        return


    # cdef void __alloc_messages(self):
    #     if self.__var_to_check is None:
    #         self.__var_to_check = <double[:self.__edge_num]>malloc(
    #             self.__edge_num*sizeof(double))
    #         if self.__var_to_check is None:
    #             raise MemoryError()
    #     if self.__check_to_var is None:
    #         self.__check_to_var = <double[:self.__edge_num]>malloc(
    #             self.__edge_num*sizeof(double))
    #         if self.__check_to_var is None:
    #             self.__free_messages()
    #             raise MemoryError()
    #     if self.__updated_lappr is None:
    #         self.__updated_lappr = <double[:self.__var_num]>malloc(
    #             self.__var_num*sizeof(double))
    #         if self.__updated_lappr is None:
    #             self.__free_messages()
    #             raise MemoryError()
    #     return

    
    # cdef void __free_messages(self):
    #     __free_array(<void*>self.__var_to_check)
    #     __free_array(<void*>self.__check_to_var)
    #     __free_array(<void*>self.__updated_lappr)
    #     self.__var_to_check  = None
    #     self.__check_to_var  = None
    #     self.__updated_lappr = None
    #     return


    
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
        # cdef long * index_set
        cdef long * vnode_set
        cdef int i
        # cdef unsigned char parity = self.__synd[check_node_index]
        cdef unsigned char parity
        with cython.boundscheck(False):
            parity = self.__synd[check_node_index]

        # index_set = self.__c_to_e[check_node_index]
        vnode_set = self.__c_to_v[check_node_index]
        # for i in range(1, index_set[0]+1):
        with cython.boundscheck(False):
            for i in range(1, vnode_set[0]+1):
                # toggle parity every time the word bit is 1
                # parity ^= self.__word[self.__e_to_v[index_set[i]]]
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
        # cdef long * index_set = self.__c_to_e[check_node_index]
        cdef long * vnode_set = self.__c_to_v[check_node_index]
        # cdef unsigned char parity = self.__synd[check_node_index]
        cdef unsigned char parity
        
        with cython.boundscheck(False):
            parity = self.__synd[check_node_index]
            
            # for i in range(1, index_set[0]+1):
            for i in range(1, vnode_set[0]+1):
                # if (self.__updated_lappr[self.__e_to_v[index_set[i]]] < 0):
                if (self.__updated_lappr[vnode_set[i]] < 0):
                    # toggle parity every time a negative lappr is met
                    parity ^= 0b1
        
        return parity ^ 0b1


    cdef unsigned char __check_lappr(self) nogil:
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
        
        
        self.__updated_lappr = lappr
        self.__synd = synd

        res = self.__check_lappr()

        self.__word = None
        self.__synd = None

        return res
    

    """ Message passing processing functions """
    cdef void __process_var_node(self, long node_index) nogil:
        cdef long* index_set
        cdef int i
        # index_set = np.array(self.edges.eid[self.edges.vid==node_index])
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
        self.__updated_lappr = updated_lappr
        
        self.__process_var_node(node_index)

        self.__lappr_data    = None
        self.__check_to_var  = None
        self.__var_to_check  = None
        self.__updated_lappr = None

        return

            
    cdef void __process_check_node(self, long node_index) nogil:
        cdef long* index_set
        cdef double prefactor
        cdef int i, j
        cdef double _atanh, _msg

        index_set = self.__c_to_e[node_index]

        with cython.boundscheck(False):
            prefactor = -2.0 if self.__synd[node_index] else 2.0
            for i in range(1, index_set[0]+1):
                _atanh = 1e300 #should be large enough
                
                for j in range(1, index_set[0]+1):
                    if (i!=j):
                        _msg = self.__var_to_check[index_set[j]]
                        _atanh = __sgn(_atanh)*__sgn(_msg) * \
                            min(cabs(_atanh), cabs(_msg)*0.5)\
                            + 0.5*(cln(1+cexp(-cabs(_msg + 2*_atanh))) -\
                                   cln(1+cexp(-cabs(_msg - 2*_atanh))))
                        
                self.__check_to_var[index_set[i]] = prefactor*_atanh
        return

    
    cpdef void process_check_node(self,
                                  long node_index,
                                  unsigned char [:] synd,
                                  double [:] check_to_var,
                                  double [:] var_to_check):
        self.__check_to_var = check_to_var
        self.__var_to_check = var_to_check
        self.__synd         = synd

        self.__process_check_node(node_index)

        self.__check_to_var = None
        self.__var_to_check = None
        self.__synd         = None
            
        return
    
        
    cpdef tuple decode(self,
                       double [:] lappr_data,
                       unsigned char [:] synd,
                       int max_iterations):

        cdef long iter_index, c, v, e
        
        if (self.check_lappr(lappr_data, synd)):
            return (True, 0, lappr_data)

        #self.__alloc_messages()
        self.__check_to_var  = np.zeros(self.__edge_num, dtype=np.double)
        self.__var_to_check  = np.empty_like(self.__check_to_var)
        self.__updated_lappr = np.empty_like(lappr_data)
        # self.__check_to_var  = cvarray(shape=(self.__edge_num,), itemsize=sizeof(double), format="i")
        # self.__var_to_check  = cvarray(shape=(self.__edge_num,), itemsize=sizeof(double), format="i")
        # self.__updated_lappr = cvarray(shape=(self.__var_num ,), itemsize=sizeof(double), format="i")
        self.__lappr_data    = lappr_data
        self.__synd          = synd

        # Initialize __check_to_var to 0
        for e in range(self.__edge_num):
            self.__check_to_var[e] = 0
        
        # First half iteration to propagate lapprs to check nodes
        # The following line also initializes var_to_check
        for v in range(self.__var_num):
            self.__process_var_node(v)

        
        for iter_index in range(max_iterations):
            for c in range(self.__chk_num):
                self.__process_check_node(c)
                
            for v in range(self.__var_num):
                self.__process_var_node(v)
                
            if (self.__check_lappr()):
                return (True, iter_index+1, self.__updated_lappr)

        return (False, max_iterations, self.__updated_lappr)


    
    
