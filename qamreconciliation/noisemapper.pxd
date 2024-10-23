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


cdef class NoiseMapper:
    cdef readonly:
        double [:] thresholds
        double [:] F_Y_thresholds
        int order
        int half_order
        int bit_per_symbol
        double [:] constellation
        double variance
        double [:,:] back_transition_probability
        double [:,:] fwrd_transition_probability
        double [:] probabilities
        double noise_var
        double noise_sigma
        double [:,:] bare_llr_table
        double [:,:] inf_erf_table
        double [:] delta_F_Y
        unsigned char [:] sign_config
        
    cdef:
        double __sigma
        
        double __y_low
        double __y_high
        double [:] _y_range
        int __n_points
        double [:] _F_Y


        int __ref_symb
        double __ref_symb_value
        double __ref_delta_F_Y
        double __ref_F_Y_threshold


    cpdef double [:] F_Y(self, double [:] y)

    cdef double _single_F_Y(self, double y)

    cpdef double g(self, double y, int i)

    cpdef double g_inv(self, double n_hat, int i)

    cpdef double g_inv_search(self, double n_hat, int i, double y_accuracy=*)

    cpdef long [:] hard_decide_index(self, double [:] y_samples)

    cpdef double [:] index_to_val(self, long [:] index)

    cpdef double [:] map_noise(self, double [:] y_samples, long [:] index)

    cpdef double [:] demap_noise(self, double [:] n_hat, long [:] symb)

    cpdef double [:] demap_noise_search(self, double [:] n_hat, long [:] symb, double y_accuracy=*)

    cpdef double [:] bare_llr(self, long [:] symb)

    # LLR construction functions
    cpdef double [:] demap_lappr(self, double n, long j)
    cpdef double [:] demap_lappr_array(self, double [:] n, long [:] j)

    cpdef double [:] demap_lappr_simplified(self, double n, long j)
    cpdef double [:] demap_lappr_simplified_array(self, double [:] n, long [:] j)
    
    cpdef double [:] demap_lappr_sofisticated(self, double n, long j)
    cpdef double [:] demap_lappr_sofisticated_array(self, double [:] n, long [:] j)
    




cdef class NoiseDemapper(NoiseMapper):
    # Keep for compatibility with simulations
    # Functionalities moved to NoiseMapper
    pass



cdef class NoiseMapperFlipSign(NoiseMapper):
    pass


cdef class NoiseMapperAntiFlipSign(NoiseMapper):
    pass
