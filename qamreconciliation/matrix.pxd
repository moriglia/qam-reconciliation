cdef class Matrix:
    cdef:
        long [:] __vnode_arr
        long [:] __cnode_arr
        int __vnode_num
        int __cnode_num
        int __edge_num

    cpdef unsigned char [:] eval_syndrome(
        self,
        unsigned char [:] word
    )
