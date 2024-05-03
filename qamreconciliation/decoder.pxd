cimport numpy as np


cdef class Decoder:
    cdef:
        long [:] __cnode_arr
        long [:] __vnode_arr
        int __edge_num
        int __vnode_num
        int __cnode_num
        long ** __c_to_e
        long ** __v_to_e
        long double [:] __check_to_var 
        long double [:] __var_to_check 
        long double [:] __updated_lappr
        # long double [:] __tanh_values  
        long double [:] __lappr_data   
        unsigned char [:] __synd
        unsigned char [:] __word    

    cdef void __free_tables(self)

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
        long double[:]    lappr,
        unsigned char [:] synd
    )


    cdef void __process_var_node(self, int node_index)


    cpdef void process_var_node(
        self,
        int             node_index,
        long double [:] lappr_data,
        long double [:] check_to_var,
        long double [:] var_to_check,
        long double [:] updated_lappr
    )


    cdef void __process_check_node(self, int node_index)


    cpdef void process_check_node(
        self,
        int               node_index,
        unsigned char [:] synd,
        long double [:]   check_to_var,
        long double [:]   var_to_check
    )


    cpdef tuple decode(
        self,
        long double [:]   lappr_data,
        unsigned char [:] synd,
        int max_iterations
    )

    
