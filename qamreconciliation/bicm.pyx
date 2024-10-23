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

# BICM conversions
import numpy as np
cimport numpy as np


cdef inline int __pow2(int i) noexcept nogil:
    return 1<<i


cpdef unsigned char [:,:] generate_table_s_to_b(int log_order):
    cdef unsigned char [:,:] res
    # cdef unsigned char [:,:] next_suborder
    cdef int half_table_index = __pow2(log_order - 1)
    
    if (log_order <= 0):
        raise ValueError(f"log_order ({log_order}) must be a positive integer")
    if (log_order==1):
        return np.array([[0], [1]], dtype=np.ubyte)
    
    res = np.empty((half_table_index<<1, log_order), dtype=np.ubyte)
    res[half_table_index:,log_order-1] = 1
    res[:half_table_index,log_order-1] = 0
    res[:half_table_index,:log_order-1] = generate_table_s_to_b(log_order-1)
    res[half_table_index:,:log_order-1] = res[half_table_index-1::-1,:log_order-1]
    return res




cpdef long [:,:] generate_error_number_table(unsigned char [:,:] s_to_b):
    cdef long [:,:] n_err = np.empty((s_to_b.shape[0], s_to_b.shape[0]), dtype=int)
    cdef int i, j, k, s


    # n_err[i, j] = number of errors when a_i is received given a_j was transmitted
    
    for i in range(s_to_b.shape[0]):
        for j in range(i):
            s = 0
            for k in range(s_to_b.shape[i]):
                s += (s_to_b[i,k] ^ s_to_b[j,k]) & 0b1;
            n_err[i, j] = s

        n_err[i,i] = 0

    for i in range(s_to_b.shape[0]):
        for j in range(i+1, s_to_b.shape[0]):
            n_err[i,j] = n_err[j, i]

    return n_err

