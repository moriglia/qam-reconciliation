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
