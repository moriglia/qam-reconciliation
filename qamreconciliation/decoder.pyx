# -*- mode: cython; gud -*- 
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free


cdef void __free_table(long** table, long tsize):
    cdef int i;
    if (table):
        for i in range(tsize):
            if (table[i]):
                free(table[i])
            else:
                break
        free(table)
    return


cdef long** __build_table(long[:] e_to_x, int num_x):
    cdef long i, e, l;
    cdef long** table;
    cdef list edges_i;

    table = <long**>malloc(sizeof(long*)*num_x);
    if not table:
        return <long**>0
    
    for i in range(num_x):
        # look for edges involving node i 
        edges_i = [];
        l = 0;
        for e in range(e_to_x.size):
            if (e_to_x[e]==i):
                # the selected edge `e` ends in node `i`
                # so, I add `e` to the list of edges of `i`
                edges_i.append(e)
                l += 1

        # store the edges in the table
        # using the first element of the table
        # entry as the size of the entry
        table[i] = <long*>malloc(sizeof(long)*(l+1))
        if not table[i]:
            __free_table(table, i+1)
            return <long**>0
        table[i][0] = l;
        for e in range(len(edges_i)):
            # now e means another thing
            # but I am just re-using it
            table[i][e+1] = edges_i[e];
    
    return table;


cdef class Decoder:
    """ Decoder

        Decoder(long [:] vnode_array, long [:] cnode_array)

        vnode_array: each entry `i` contains the index of the
            variable node to which the `i`-th edge is connected

        cnode_array: each entry `i` contains the index of the
            check node to which the `i`-th edge is connected

        that means, edge `i`:
            (cnode_array[i], cnode_array[i])
        connects check node cnode_array[i] to var node vnode_array[i]
    """
    def __cinit__(self, long [:] vnode_array, long [:] cnode_array):
        if (vnode_array.shape[0] != cnode_array.shape[0]):
            raise ValueError("Incompatible sizes for input vectors")
        
        # store arrays
        self.__vnode_arr = vnode_array
        self.__cnode_arr = cnode_array
        
        # get edge, cnode and vnode numbers
        self.__edge_num  = vnode_array.shape[0]
        self.__cnode_num = max(cnode_array) + 1
        self.__vnode_num = max(vnode_array) + 1

        ###############################################
        ###      Evaluate node to edge tables:      ###
        ###############################################
        self.__c_to_e = __build_table(cnode_array, self.__cnode_num)
        if not self.__c_to_e:
            raise MemoryError()
        self.__v_to_e = __build_table(vnode_array, self.__vnode_num)
        if not self.__v_to_e:
            __free_table(self.__c_to_e, self.__c_num)
            raise MemoryError()
        
        # self.__c_to_e = <long**>malloc(sizeof(long*)*self.__cnode_num)
        # if not self.__c_to_e:
        #     raise MemoryError()
        # self.__v_to_e = <long**>malloc(sizeof(long*)*self.__vnode_num)
        # if not self.__v_to_e:
        #     free(self.__c_to_e)
        #     raise MemoryError()

        # cdef list tmp_edges
        # cdef long i, j, l
        # for i in range(self.__cnode_num):
        #     tmp_edges = []
        #     l = 0
            
        #     # save all edge indices in a list
        #     for j in range(self.__edge_num):
        #         if (self.__cnode_arr[j]==i):
        #             tmp_edges.append(j)
        #             l+=1
                    
        #     # allocate the right size for the list of coefficients
        #     # and store them in the new array
        #     self.__c_to_e[i] = <long*>malloc(sizeof(long)*(l+1))
        #     if not self.__c_to_e[i]:
        #         for j in range(i):
        #             free(self.__c_to_e[j])
        #         free(self.__c_to_e)
        #         free(self.__v_to_e)
        #         raise MemoryError()
        #     self.__c_to_e[i][0] = l # first element is the degree of the check node
        #     for j in range(l):
        #         self.__c_to_e[i][j+1] = tmp_edges[j]

        # # repeat for variable nodes
        # for i in range(self.__vnode_num):
        #     tmp_edges = []
        #     l = 0
            
        #     # save all edge indices in a list
        #     for j in range(self.__edge_num):
        #         if (self.__vnode_arr[j]==i):
        #             tmp_edges.append(j)
        #             l += 1
                    
        #     # allocate the right size for the list of coefficients
        #     # and store them in the new array
        #     self.__v_to_e[i] = <long*>malloc(sizeof(long)*(l+1))
        #     if not self.__v_to_e[i]:
        #         for j in range(i):
        #             free(self.__v_to_e[j])
        #         for j in range(self.__cnode_num):
        #             free(self.__c_to_e[j])
        #         free(self.__v_to_e)
        #         free(self.__c_to_e)
        #         raise MemoryError()
        #     self.__v_to_e[i][0] = l # first element is the degree of the check node
        #     for j in range(l):
        #         self.__v_to_e[i][j+1] = tmp_edges[j]

        return

    
    cdef void __free_tables(self):
        __free_table(self.__v_to_e, self.__vnode_num)
        __free_table(self.__c_to_e, self.__cnode_num)
        return

    
    def __dealloc__(self):
        self.__free_tables()
        return
    
    
    @property
    def cnum(self) -> int:
        """ Number of check nodes """
        return self.__cnode_num

    
    @property
    def vnum(self) -> int:
        """ Number of variable nodes """
        return self.__vnode_num


    @property
    def ednum(self) -> int:
        """ Number of edges """
        return self.__edge_num


    """ Syndrome checking functions """
    cdef unsigned char __check_synd_node(self, int check_node_index):
        cdef unsigned char check_sum = self.__synd[check_node_index] ^ 0b1
        # so that check_sum==synd[...] is true
        cdef int i
        for i in range(1, self.__c_to_e[check_node_index][0]+1):
            check_sum ^= self.__word[self.__vnode_arr[self.__c_to_e[check_node_index][i]]]
        return check_sum


    cpdef unsigned char check_synd_node(
        self,
        int               check_node_index,
        unsigned char [:] word,
        unsigned char [:] synd):
        self.__word = word
        self.__synd = synd

        try:
            return self.__check_synd_node(check_node_index)
        finally:
            self.__word = None
            self.__synd = None


    
    cdef unsigned char __check_word(self):
        cdef int i;
        for i in range(self.__cnode_num):
            if (self.__check_synd_node(i) == 0):
                return 0
        return 1
    

    cpdef unsigned char check_word(
        self,
        unsigned char [:] word,
        unsigned char [:] synd
    ):
        self.__word = word
        self.__synd = synd

        try:
            return self.__check_word()
        finally:
            self.__word = None
            self.__synd = None


    cdef unsigned char __check_synd_node_lappr(self, int node_index):
        cdef int i ;
        cdef unsigned char parity = 0
        # -1.0 if (self.__synd[node_index]) else 1.0 ;
        for i in range(1, self.__c_to_e[node_index][0]+1):
            if (self.__updated_lappr[self.__vnode_arr[self.__c_to_e[node_index][i]]] == 0):
                return 0
            if (self.__updated_lappr[self.__vnode_arr[self.__c_to_e[node_index][i]]] < 0):
                parity ^= 0b1
            # prod *= self.__updated_lappr[self.__vnode_arr[self.__c_to_e[node_index][i]]]
        parity ^= self.__synd[node_index] ^ 0b1
        return parity & 0b1


    cdef unsigned char __check_lappr(self):
        cdef int i;
        for i in range(self.__cnode_num):
            if (self.__check_synd_node_lappr(i) == 0):
                return 0
        return 1


    cpdef unsigned char check_lappr(
        self,
        long double [:]   lappr,
        unsigned char [:] synd
    ):
        self.__updated_lappr = lappr
        self.__synd = synd

        try:
            return self.__check_lappr()
        finally:
            self.__updated_lappr = None
            self.__synd = None
    

    """ Message passing processing functions """
    cdef void __process_var_node(self, int node_index):
        # cdef np.ndarray[long, ndim=1] index_set = \
        #     np.nonzero(np.asarray(self.__vnode_arr, dtype=int)==node_index)[0]

        cdef int i
        self.__updated_lappr[node_index] = self.__lappr_data[node_index]
        for i in range(1, self.__v_to_e[node_index][0]+1):
            self.__updated_lappr[node_index] += self.__check_to_var[self.__v_to_e[node_index][i]]

        
        for i in range(1, self.__v_to_e[node_index][0]+1):
            self.__var_to_check[self.__v_to_e[node_index][i]] = \
                self.__updated_lappr[node_index] - \
                self.__check_to_var[self.__v_to_e[node_index][i]]
        return


    cpdef void process_var_node(
        self,
        int node_index,
        long double [:] lappr_data,
        long double [:] check_to_var,
        long double [:] var_to_check,
        long double [:] updated_lappr
    ):
        self.__lappr_data = lappr_data
        self.__check_to_var = check_to_var
        self.__var_to_check = var_to_check
        self.__updated_lappr = updated_lappr
        
        try:
            self.__process_var_node(node_index)
        finally:
            self.__lappr_data    = None
            self.__check_to_var  = None
            self.__var_to_check  = None
            self.__updated_lappr = None

        return

            
    cdef void __process_check_node(self, int node_index):
        cdef long double prefactor
        cdef long double arctanh_partial# = 1.0
        cdef long double message_from_v
        cdef int i, j #, zerocount, zeroindex
        # zerocount = 0
        prefactor = -2.0 if (self.__synd[node_index]==0b1) else 2.0
        # for i in range(1, self.__c_to_e[node_index][0]+1):
        #     if (self.__tanh_values[self.__c_to_e[node_index][i]]==0.0):
        #         zerocount += 1
        #         zeroindex = i
        #     else:
        #         allprod *= self.__tanh_values[self.__c_to_e[node_index][i]]
            
        # if (zerocount == 0):
        #     for i in range(1, self.__c_to_e[node_index][0]+1):
        #         self.__check_to_var[self.__c_to_e[node_index][i]] = prefactor*np.arctanh(
        #             allprod / self.__tanh_values[self.__c_to_e[node_index][i]]
        #         )
        # else:
        #     if (zerocount != 1):
        #         zeroindex = -1
        #     for i in range(1, self.__c_to_e[node_index][0]+1):
        #         if (zeroindex==i):
        #             self.__check_to_var[self.__c_to_e[node_index][i]] = prefactor*np.arctanh(allprod)
        #         else:
        #             self.__check_to_var[self.__c_to_e[node_index][i]] = 0.0
        for i in range(1, self.__c_to_e[node_index][0]+1):
            arctanh_partial = 0.5*self.__var_to_check[self.__c_to_e[node_index][1]]
            for j in range(2, self.__c_to_e[node_index][0]+1):
                if (i!=j):
                    message_from_v = self.__var_to_check[self.__c_to_e[node_index][j]]
                    arctanh_partial = \
                        np.sign(arctanh_partial)*np.sign(message_from_v)*\
                        min(np.abs(arctanh_partial),
                            np.abs(message_from_v)*0.5) + \
                        0.5*(np.log(1+np.exp(-np.abs(message_from_v+2*arctanh_partial))) - \
                             np.log(1+np.exp(-np.abs(message_from_v-2*arctanh_partial))))
                    #allprod *= self.__tanh_values[self.__c_to_e[node_index][j]]
            self.__check_to_var[self.__c_to_e[node_index][i]] = prefactor*arctanh_partial

        return
                
    
    cpdef void process_check_node(
        self,
        int node_index,
        unsigned char [:] synd,
        long double [:] check_to_var,
        long double [:] var_to_check):

        cdef long i
        
        self.__check_to_var = check_to_var
        self.__var_to_check = var_to_check
        self.__synd         = synd
        # self.__tanh_values  = np.empty_like(var_to_check)

        # for i in range(len(var_to_check)):
        #     self.__tanh_values[i] = np.tanh(0.5*var_to_check[i])
            
        try:
            self.__process_check_node(node_index)
        finally:
            self.__check_to_var = None
            self.__var_to_check = None
            self.__synd         = None
            # self.__tanh_values  = None
            
        return
    
        
    cpdef tuple decode(
        self,
        long double [:] lappr_data,
        unsigned char [:] synd,
        int max_iterations
    ) :
        cdef int iter_index, v, c, e
        # cdef unsigned char check_ok

        # if (self.check_lappr(lappr_data, synd)):
        #     return (True, 0, lappr_data)

        self.__check_to_var  = np.zeros(self.ednum, dtype=np.longdouble)
        self.__var_to_check  = np.empty_like(self.__check_to_var)
        self.__updated_lappr = np.empty_like(lappr_data)
        # self.__tanh_values   = np.empty_like(self.__var_to_check)
        self.__lappr_data    = lappr_data
        self.__synd          = synd
        
        try:
            if (self.__check_lappr()):
                return (True, 0, lappr_data)

            # First half iteration to propagate lapprs to check nodes
            # The following line also initializes var_to_check
            for v in range(self.vnum):
                self.__process_var_node(v)
            
            
            for iter_index in range(1,max_iterations+1):
                # prepare tanh values for check node processing
                # for e in range(self.__edge_num):
                #     self.__tanh_values[e] = np.tanh(0.5*self.__var_to_check[e])

                # check node processing (half iteration)
                # check_ok = 1
                for c in range(self.__cnode_num):
                    # check_ok *= self.__process_check_node(c)
                    self.__process_check_node(c)

                # variable node processing (second half iteration)
                for v in range(self.__vnode_num):
                    self.__process_var_node(v)

                # check for completion
                if (self.__check_lappr()):
                    return (True, iter_index, self.__updated_lappr)

            return (False, max_iterations, self.__updated_lappr)
        
        finally:
            self.__check_to_var  = None
            self.__var_to_check  = None
            self.__updated_lappr = None
            self.__lappr_data    = None
            self.__synd          = None
            # self.__tanh_values   = None

