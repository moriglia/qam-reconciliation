cdef class Matrix:
    cdef readonly:
        long [:] vnode_arr
        long [:] cnode_arr
        int vnode_num
        int cnode_num
        int edge_num

    cpdef unsigned char [:] eval_syndrome(
        self,
        unsigned char [:] word
    )
