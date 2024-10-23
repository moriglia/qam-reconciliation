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
cdef struct decoderResult:
    unsigned char   success
    int             iterations
    double        * final_lappr
    long            final_lappr_size
    

cdef class Decoder:
    cdef:
        long __chk_num, __var_num, __edge_num
        long [:] __e_to_c
        long [:] __e_to_v
        long ** __c_to_e
        long ** __v_to_e
        long ** __c_to_v
        double ** __cnode_buffer_list
        
        double [:] __lappr_data
        double  *  __updated_lappr
        double [:] __check_to_var
        double [:] __var_to_check

        unsigned char [:] __synd
        unsigned char [:] __word

    
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

    cdef unsigned char __check_lappr(self) noexcept nogil

    cpdef unsigned char check_lappr(self,
                                    double [:] lappr,
                                    unsigned char [:] synd)

    cdef void __process_var_node(self, long node_index) noexcept nogil

    cpdef void process_var_node(self,
                                long node_index,
                                double [:] lappr_data,
                                double [:] check_to_var,
                                double [:] var_to_check,
                                double [:] updated_lappr)

    cdef int __process_check_node(self, long node_index) noexcept nogil

    cpdef int process_check_node(self,
                                 long node_index,
                                 unsigned char [:] synd,
                                 double [:] check_to_var,
                                 double [:] var_to_check)

    cdef (int, int) _decode(self,
                            double [:] lappr_data,
                            unsigned char [:] synd,
                            int max_iterations,
                            double [:] final_lappr) noexcept nogil
    
    cpdef tuple decode(self,
                       double [:] lappr_data,
                       unsigned char [:] synd,
                       int max_iterations)
