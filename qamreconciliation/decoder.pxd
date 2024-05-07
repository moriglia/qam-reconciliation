cdef class Decoder:
    cdef:
        long __chk_num, __var_num, __edge_num
        long [:] __e_to_c
        long [:] __e_to_v
        long ** __c_to_e
        long ** __v_to_e
        long ** __c_to_v
        
        double [:] __lappr_data
        double [:] __updated_lappr
        double [:] __check_to_var
        double [:] __var_to_check

        unsigned char [:] __synd
        unsigned char [:] __word

    # cdef void __alloc_messages(self)

    # cdef void __free_messages(self)
        
    cdef unsigned char __check_synd_node(self, long check_node_index) noexcept nogil

    cpdef unsigned char check_synd_node(self,
                                        long check_node_index,
                                        unsigned char [:] word,
                                        unsigned char [:] synd)
    
    cdef unsigned char __check_word(self) nogil

    cpdef unsigned char check_word(self,
                                   unsigned char [:] word,
                                   unsigned char [:] synd)

    cdef unsigned char __check_lappr_node(self, long check_node_index) noexcept nogil

    cdef unsigned char __check_lappr(self) nogil

    cpdef unsigned char check_lappr(self,
                                    double [:] lappr,
                                    unsigned char [:] synd)

    cdef void __process_var_node(self, long node_index) nogil

    cpdef void process_var_node(self,
                                long node_index,
                                double [:] lappr_data,
                                double [:] check_to_var,
                                double [:] var_to_check,
                                double [:] updated_lappr)

    cdef void __process_check_node(self, long node_index) nogil

    cpdef void process_check_node(self,
                                  long node_index,
                                  unsigned char [:] synd,
                                  double [:] check_to_var,
                                  double [:] var_to_check)

    cpdef tuple decode(self,
                       double [:] lappr_data,
                       unsigned char [:] synd,
                       int max_iterations)
