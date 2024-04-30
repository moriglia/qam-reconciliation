# -*- mode: cython; gud -*- 
import numpy as np
cimport numpy as np
from functools import reduce


cdef class Decoder:
    """ Decoder

        Decoder(edge_data)
    
        edge_data: pandas.DataFrame with columns
        - 'eid' (edge ID)
        - 'cid' (check node ID)
        - 'vid' (variable node ID)
    
    """

    def __init__(self, edge_data:pd.DataFrame, num_data_first_row = True):
        if (num_data_first_row):
            self.__edge_arr = edge_data.eid[1:].to_numpy()
            self.__cnode_arr= edge_data.cid[1:].to_numpy()
            self.__vnode_arr= edge_data.vid[1:].to_numpy()
            self.__edge_num = edge_data['eid'][0]
            self.__chk_num  = edge_data['cid'][0]
            self.__var_num  = edge_data['vid'][0]
        else:
            self.__edge_arr = edge_data.eid.to_numpy()
            self.__cnode_arr= edge_data.cid.to_numpy()
            self.__vnode_arr= edge_data.vid.to_numpy()
            self.__edge_num = len(edge_data)
            self.__chk_num  = edge_data['cid'].max() + 1
            self.__var_num  = edge_data['vid'].max() + 1
        return
    
    @property
    def cnum(self) -> int:
        """ Number of check nodes """
        return self.__chk_num

    
    @property
    def vnum(self) -> int:
        """ Number of variable nodes """
        return self.__var_num


    @property
    def ednum(self) -> int:
        """ Number of edges """
        return self.__edge_num


    """ Syndrome checking functions """
    cdef unsigned char __check_synd_node(self, int check_node_index):
        cdef int check_sum = 1-self.__synd[check_node_index]
        cdef int i
        for i in range(self.__edge_num):
            if (self.__cnode_arr[i] == check_node_index):
                check_sum += self.__word[self.__vnode_arr[i]]
        return check_sum & 0b1 # equivalent to `check_sum % 2`


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
        cdef int _sum = 0;
        for i in range(self.__chk_num):
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
        cdef double prod = -1.0 if (self.__synd[node_index]) else 1.0 ;
        for i in range(self.__edge_num):
            if (self.__cnode_arr[i] == node_index):
                prod *= self.__updated_lappr[self.__vnode_arr[i]]
        return prod > 0.0


    cdef unsigned char __check_lappr(self):
        cdef int i;
        for i in range(self.__chk_num):
            if (self.__check_synd_node_lappr(i) == 0):
                return 0
        return 1


    cpdef unsigned char check_lappr(
        self,
        double [:]        lappr,
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
        cdef np.ndarray[long, ndim=1] index_set = \
            np.nonzero(np.asarray(self.__vnode_arr, dtype=int)==node_index)[0]

        cdef int i
        self.__updated_lappr[node_index] = self.__lappr_data[node_index]
        for i in range(len(index_set)):
            self.__updated_lappr[node_index] += self.__check_to_var[index_set[i]]

        
        for i in range(len(index_set)):
            self.__var_to_check[index_set[i]] = \
                self.__updated_lappr[node_index] - \
                self.__check_to_var[index_set[i]]
        return


    cpdef void process_var_node(
        self,
        int node_index,
        double [:] lappr_data,
        double [:] check_to_var,
        double [:] var_to_check,
        double [:] updated_lappr
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
        cdef double prefactor
        cdef double allprod = 1.0
        cdef int i
        cdef np.ndarray[long, ndim=1] index_set = \
            np.nonzero(np.asarray(self.__cnode_arr, dtype=int)==node_index)[0]
        prefactor = -2.0 if self.__synd[node_index] else 2.0
        for i in range(len(index_set)):
            allprod*= self.__tanh_values[index_set[i]]
            
        if (allprod != 0):
            for i in range(len(index_set)):
                self.__check_to_var[index_set[i]] = prefactor*np.arctanh(
                    allprod / self.__tanh_values[index_set[i]]
                )
        else:
            for i in range(len(index_set)):
                self.__check_to_var[index_set[i]] = prefactor*np.arctanh(
                    reduce(lambda x, y: x*y,
                           map(lambda x: np.tanh(0.5*x),
                               [*np.array(self.__var_to_check[index_set[   :i]]),
                                *np.array(self.__var_to_check[index_set[i+1: ]])])))

        return

    
    cpdef void process_check_node(
        self,
        int node_index,
        unsigned char [:] synd,
        double [:] check_to_var,
        double [:] var_to_check):

        cdef long i
        
        self.__check_to_var = check_to_var
        self.__var_to_check = var_to_check
        self.__synd         = synd
        self.__tanh_values  = np.empty_like(var_to_check)

        for i in range(len(var_to_check)):
            self.__tanh_values[i] = np.tanh(0.5*var_to_check[i])
            
        try:
            self.__process_check_node(node_index)
        finally:
            self.__check_to_var = None
            self.__var_to_check = None
            self.__synd         = None
            self.__tanh_values  = None
            
        return
    
        
    cpdef tuple decode(
        self,
        double [:] lappr_data,
        unsigned char [:] synd,
        int max_iterations
    ) :
        cdef int iter_index, v, c, e

        if (self.check_lappr(lappr_data, synd)):
            return (True, 0, lappr_data)

        self.__check_to_var  = np.zeros(self.ednum, dtype=np.double)
        self.__var_to_check  = np.empty_like(self.__check_to_var)
        self.__updated_lappr = np.empty_like(lappr_data)
        self.__tanh_values   = np.empty_like(self.__var_to_check)
        self.__lappr_data    = lappr_data
        self.__synd          = synd
        
        try:
            # First half iteration to propagate lapprs to check nodes
            # The following line also initializes var_to_check
            for v in range(self.vnum):
                self.__process_var_node(v)
            
            
            for iter_index in range(1,max_iterations+1):
                # prepare tanh values for check node processing
                for e in range(self.__edge_num):
                    self.__tanh_values[e] = np.tanh(0.5*self.__var_to_check[e])
                # check node processing (half iteration)
                for c in range(self.__chk_num):
                    self.__process_check_node(c)

                # variable node processing (second half iteration)
                for v in range(self.__var_num):
                    self.__process_var_node(v)

                # check for completion
                if (self.__check_lappr()):
                    return (True, iter_index+1, self.__updated_lappr)

            return (False, max_iterations, self.__updated_lappr)
        
        finally:
            self.__check_to_var  = None
            self.__var_to_check  = None
            self.__updated_lappr = None
            self.__lappr_data    = None
            self.__synd          = None
            self.__tanh_values   = None

