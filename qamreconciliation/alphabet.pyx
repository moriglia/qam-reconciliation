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
cimport numpy as np
import numpy as np
# from galois import GF2
from . cimport bicm
from cython.view cimport array as cvarray




cdef class Alphabet:
    pass
    # cdef readonly:
    #     double [:] constellation
    #     double [:] thresholds
    #     double variance
    #     int order


cdef class PAMAlphabet(Alphabet):
    def __cinit__(self, unsigned char bit_per_symbol, double step, double[:] probabilities=None):
        cdef int i
        cdef double tmp
        
        if (bit_per_symbol==0):
            raise ValueError(f"Bit per symbol must be at least 1, got {bit_per_symbol}")

        self.bit_per_symbol = bit_per_symbol
        self.order = 1<<bit_per_symbol
        self.step = step

        if probabilities is None:
            self.probabilities = np.ones(self.order, dtype=np.double)/self.order
        elif probabilities.size != self.order:
            raise ValueError(f"Probability vector does not match constellation size")
        else:
            tmp = 0
            for i in range(probabilities.size):
                if probabilities[i] <= 0:
                    ValueError(f"Probabilities must be positive")
                tmp += probabilities[i]

            if np.abs(tmp - 1) > 1e-9:
                raise ValueError(f"Probabilities do not sum to 1")
            self.probabilities = probabilities
        
        
        self.constellation = (np.arange(self.order)-(self.order-1)/2)*step
        self.variance = 0
        self.thresholds = np.empty(self.order+1, dtype=np.double)

        for i in range(0, self.order):
            self.variance += self.probabilities[i] * np.abs(self.constellation[i])**2
        
        for i in range(1, self.order):
            self.thresholds[i] = \
                self.constellation[i] - step/2
        self.thresholds[0] = self.constellation[0]*100  # is negative
        self.thresholds[-1]= self.constellation[-1]*100 # is positive
        
        self.s_to_b = bicm.generate_table_s_to_b(self.bit_per_symbol)
        return


    cpdef long [:] random_symbols(self, int N):
        return np.array(
            np.random.choice(self.order, size=N, p=self.probabilities),
            dtype=int
        )


    cpdef double [:] index_to_value(self, long [:] index):
        cdef int i
        cdef double [:] vals = cvarray(shape=(index.size,),
                                       itemsize=sizeof(double),
                                       format="d")
        
        for i in range(index.size):
            vals[i] = self.constellation[index[i]]

        return vals


    cpdef unsigned char [:] demap_symbols_to_bits(self, long [:] symbol_index):
        cdef long i
        cdef unsigned char [:] bit_array = cvarray(shape=(symbol_index.size*self.bit_per_symbol,),
                                                   itemsize=sizeof(unsigned char),
                                                   format="c")

        for i  in range(symbol_index.size):
            bit_array[i*self.bit_per_symbol:(i+1)*self.bit_per_symbol] = self.s_to_b[symbol_index[i]]

        return bit_array
        
