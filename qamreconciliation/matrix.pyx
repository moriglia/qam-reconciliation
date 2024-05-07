cimport numpy as np
import numpy as np


cdef class Matrix:
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

    # def __init__(self, edge_data:pd.DataFrame, num_data_first_row:bool=True):
    #     if (num_data_first_row):
    #         self.__edge_df  = edge_data[:][1:]
    #         self.__edge_num = edge_data['eid'][0]
    #         self.__chk_num  = edge_data['cid'][0]
    #         self.__var_num  = edge_data['vid'][0]
    #     else:
    #         self.__edge_df  = edge_data
    #         self.__edge_num = len(edge_data)
    #         self.__chk_num  = edge_data['cid'].max() + 1
    #         self.__var_num  = edge_data['vid'].max() + 1
    #     return


    @property
    def cnum(self):
        """ Number of check nodes """
        return self.__cnode_num

    
    @property
    def vnum(self):
        """ Number of variable nodes """
        return self.__vnode_num


    @property
    def enum(self):
        """ Number of edges """
        return self.__edge_num



    """ Eval syndrome of a given word """
    cpdef unsigned char [:] eval_syndrome(self, unsigned char [:] word):
        cdef int e
        cdef unsigned char [:] synd = np.zeros(self.__cnode_num, dtype=np.ubyte)
        for e in range(self.__edge_num):
            synd[self.__cnode_arr[e]] ^= word[self.__vnode_arr[e]]
        return synd

    
