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


cdef class Alphabet:
    cdef readonly:
        double [:] constellation
        double [:] thresholds
        double variance
        int order
        double step
    


cdef class PAMAlphabet(Alphabet):
    cdef readonly:
        unsigned char bit_per_symbol
        unsigned char [:,:] s_to_b
        double [:] probabilities

        
    cpdef long [:] random_symbols(self, int N)


    cpdef double [:] index_to_value(self, long[:] index)

    cpdef unsigned char [:] demap_symbols_to_bits(self, long [:] symbol_index)
    
