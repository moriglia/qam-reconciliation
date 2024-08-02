cimport numpy as np
import numpy as np


cdef class Matrix:
    def __cinit__(self, long [:] vnode_array, long [:] cnode_array):
        if (vnode_array.shape[0] != cnode_array.shape[0]):
            raise ValueError("Incompatible sizes for input vectors")
        
        # store arrays
        self.vnode_arr = vnode_array
        self.cnode_arr = cnode_array
        
        # get edge, cnode and vnode numbers
        self.edge_num  = vnode_array.shape[0]
        self.cnode_num = max(cnode_array) + 1
        self.vnode_num = max(vnode_array) + 1

    # def __init__(self, edge_data:pd.DataFrame, num_data_first_row:bool=True):
    #     if (num_data_first_row):
    #         self.edge_df  = edge_data[:][1:]
    #         self.edge_num = edge_data['eid'][0]
    #         self.chk_num  = edge_data['cid'][0]
    #         self.var_num  = edge_data['vid'][0]
    #     else:
    #         self.edge_df  = edge_data
    #         self.edge_num = len(edge_data)
    #         self.chk_num  = edge_data['cid'].max() + 1
    #         self.var_num  = edge_data['vid'].max() + 1
    #     return


    @property
    def cnum(self):
        """ Number of check nodes """
        return self.cnode_num

    
    @property
    def vnum(self):
        """ Number of variable nodes """
        return self.vnode_num


    @property
    def enum(self):
        """ Number of edges """
        return self.edge_num



    """ Eval syndrome of a given word """
    cpdef unsigned char [:] eval_syndrome(self, unsigned char [:] word):
        cdef int e
        cdef unsigned char [:] synd = np.zeros(self.cnode_num, dtype=np.ubyte)
        for e in range(self.edge_num):
            synd[self.cnode_arr[e]] ^= word[self.vnode_arr[e]]
        return synd

    
