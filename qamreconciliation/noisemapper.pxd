

cdef class NoiseMapper:
    cdef readonly:
        double [:] thresholds
        double [:] F_Y_thresholds
        int order
        int bit_per_symbol
        double [:] constellation
        double variance
        double [:,:] back_transition_probability
        double [:,:] fwrd_transition_probability
        double [:] probabilities
        double noise_var
        double [:,:] bare_llr_table
        
    cdef:
        double __sigma
        
        double __y_low
        double __y_high
        double [:] __y_range
        int __n_points
        double [:] __F_Y
        double [:] delta_F_Y


        int __ref_symb
        double __ref_symb_value
        double __ref_delta_F_Y
        double __ref_F_Y_threshold


    cpdef double [:] F_Y(self, double [:] y)

    cdef double __single_F_Y(self, double y)

    cdef double g(self, double y, int i)

    cdef double g_inv(self, double n_hat, int i)

    cpdef long [:] hard_decide_index(self, double [:] y_samples)

    cpdef double [:] index_to_val(self, long [:] index)

    cpdef double [:] map_noise(self, double [:] y_samples, long [:] index)

    cpdef double [:] demap_noise(self, double [:] n_hat, long [:] symb)

    cpdef double [:] bare_llr(self, long [:] symb)





cdef class NoiseDemapper(NoiseMapper):
    cdef readonly double [:,:] inf_erf_table
    
    cpdef double [:] demap_lappr(self, double n, long j)
    cpdef double [:] demap_lappr_array(self, double [:] n, long [:] j)

    cpdef double [:] demap_lappr_simplified(self, double n, long j)
    cpdef double [:] demap_lappr_simplified_array(self, double [:] n, long [:] j)
    

    cpdef double [:] demap_lappr_sofisticated(self, double n, long j)
    cpdef double [:] demap_lappr_sofisticated_array(self, double [:] n, long [:] j)
    
