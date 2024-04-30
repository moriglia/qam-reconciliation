cimport numpy as np


cdef class Decoder:
    cdef:
        long [:] __edge_arr
        long [:] __cnode_arr
        long [:] __vnode_arr
        int __edge_num
        int __var_num
        int __chk_num
        double [:] __check_to_var 
        double [:] __var_to_check 
        double [:] __updated_lappr
        double [:] __tanh_values  
        double [:] __lappr_data   
        unsigned char [:] __synd
        unsigned char [:] __word    
    

    cdef unsigned char __check_synd_node(self, int check_node_index)
    

    cpdef unsigned char check_synd_node(
        self,
        int check_node_index,
        unsigned char [:] word,
        unsigned char [:] synd
    )


    cdef unsigned char __check_word(self)


    cpdef unsigned char check_word(
        self,
        unsigned char [:] word,
        unsigned char [:] synd
    )


    cdef unsigned char __check_synd_node_lappr(self, int node_index)


    cdef unsigned char __check_lappr(self)


    cpdef unsigned char check_lappr(
        self,
        double[:]         lappr,
        unsigned char [:] synd
    )


    cdef void __process_var_node(self, int node_index)


    cpdef void process_var_node(
        self,
        int                        node_index,
        double [:] lappr_data,
        double [:] check_to_var,
        double [:] var_to_check,
        double [:] updated_lappr
    )


    cdef void __process_check_node(self, int node_index)


    cpdef void process_check_node(
        self,
        int                        node_index,
        unsigned char [:] synd,
        double [:]        check_to_var,
        double [:]        var_to_check
    )


    cpdef tuple decode(
        self,
        double [:]        lappr_data,
        unsigned char [:] synd,
        int max_iterations
    )

    
