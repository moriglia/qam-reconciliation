


cdef class Alphabet:
    cdef readonly:
        double [:] constellation
        double [:] thresholds
        double variance
        int order
    


cdef class PAMAlphabet(Alphabet):
    cdef readonly:
        unsigned char bit_per_symbol
        unsigned char [:,:] s_to_b
        double [:] probabilities

        
    cdef long [:] _random_symbols(self, int N) nogil
    cdef double [:] _index_to_value(self, long[:] index) nogil
    cdef unsigned char [:] _demap_symbols_to_bits(self, long [:] symbol_index) nogil
    cpdef long [:] random_symbols(self, int N)
    cpdef double [:] index_to_value(self, long[:] index)
    cpdef unsigned char [:] demap_symbols_to_bits(self, long [:] symbol_index)
    
