# SPDX-License-Identifier: GPL-3.0-or-later
#     Copyright (C) 2024  Marco Origlia
# 
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
# 
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
# 
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
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

        self.vnum = self.__vnode_num
        self.cnum = self.__cnode_num
        self.ednum = self.__edge_num

        return

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


    """ Eval syndrome of a given word """
    cpdef unsigned char [:] eval_syndrome(self, unsigned char [:] word):
        cdef int e
        cdef unsigned char [:] synd = np.zeros(self.__cnode_num, dtype=np.ubyte)
        for e in range(self.__edge_num):
            synd[self.__cnode_arr[e]] ^= word[self.__vnode_arr[e]]
        return synd

    
