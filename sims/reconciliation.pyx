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
# cython: boundscheck=False
from qamreconciliation cimport Decoder, Matrix, NoiseMapper, PAMAlphabet
from qamreconciliation.utils cimport count_errors_from_lappr
from libc.math cimport sqrt, log, exp
import numpy as np
from cython.view cimport array as cvarray
from libc.stdlib cimport malloc, free


cdef void _y_to_lappr_grey(double y, double * constellation, int BPS, int ORD, double twoVariance, double * lappr, double * buff) noexcept nogil:
    cdef int i, l, mod_index
    cdef double * _N
    cdef double * _D
    cdef double addendum

    _N = lappr
    _D = buff
    
    for l in range(BPS):
        _N[l] = 0
        _D[l] = 0
    
    for i in range(ORD):
        addendum = exp(-((y-constellation[i])**2)/twoVariance)
        mod_index = i
        for l in range(BPS):
            if ((mod_index*(mod_index+1)) & 0b11):
                _D[l] += addendum
            else:
                _N[l] += addendum
            mod_index >>= 1
        
    for l in range(BPS):
        lappr[l] = log(_N[l]) - log(_D[l])

    return


cdef void _y_to_lappr_grey_array(double * y, long S, double * constellation, int BPS, int ORD, double twoVariance,
                                 double * lappr, double * buff = NULL) noexcept nogil:
    cdef long i, lappr_offset
    cdef double * buff_ptr
    if (buff is NULL):
        buff_ptr = <double*>malloc(sizeof(double)*S*BPS)
    else:
        buff_ptr = buff
    
    
    for i in range(S):
        lappr_offset = BPS*i
        _y_to_lappr_grey(y[i], constellation, BPS, ORD,
                         twoVariance, lappr + lappr_offset, buff_ptr + lappr_offset)

    if (buff is NULL):
        free(buff_ptr)
    
    return


cpdef double [:] y_to_lappr_grey_array(double [:] y, PAMAlphabet pa, double twoVariance)noexcept nogil:
    cdef long S, BPS, ORD
    cdef double [:] lappr
    with gil:
        S = y.size
        BPS = pa.bit_per_symbol
        ORD = pa.order
        lappr = cvarray(shape    = (S*BPS,),
                        itemsize = sizeof(double),
                        format   = "d")

    _y_to_lappr_grey_array(&y[0], S,
                           &pa.constellation[0], BPS, ORD,
                           twoVariance, &lappr[0])
    return lappr
    


cpdef tuple simulate_softening_snr_dB(double snr_dB, Decoder dec, Matrix mat,
                                      PAMAlphabet pa, unsigned char [:] nmconfig,
                                      int decoder_iterations, int simulation_loops,
                                      int ferr_count_min, double alpha=1.0):

    cdef long err_count, frame_error_count, decoding_iterations, successful_decoding,
    cdef long N, K, N_symb, wordcount, new_errors
    cdef long [:] x
    cdef double [:] y
    cdef long [:] x_hat
    cdef double [:] n_hat
    cdef double [:] lappr
    cdef double [:] final_lappr
    cdef unsigned char success
    cdef int iterations, i

    Es = pa.variance
    N0 = Es * ( 10**(-snr_dB/10) ) / 2
    
    nm = NoiseMapper(pa, N0, nmconfig)
    
    err_count = 0
    frame_error_count = 0
    
    decoding_iterations = 0
    successful_decoding = 0

    N = mat.vnum
    K = N - mat.cnum
    
    N_symb = N // pa.bit_per_symbol

    final_lappr = cvarray(shape=(N,), itemsize=sizeof(double), format="d")
    
    for wordcount in range(simulation_loops):
        # Alice generates tx symbols:
        x     = pa.random_symbols(N_symb)
        
        # Channel adds AWGN
        y     = np.array(pa.index_to_value(x)) + nm.noise_sigma*np.random.randn(N_symb)
        
        # Bob decides for the symbols based on the thresholds and builds the softening metric
        x_hat = nm.hard_decide_index(y)
        n_hat = nm.map_noise(y, x_hat)
        
        word  = pa.demap_symbols_to_bits(x_hat)
        synd  = mat.eval_syndrome(word)

        # Alice uses the transformed noise and the syndrome received from Bob
        # to build the LLR and proceed with reconciliation
        lappr = nm.demap_lappr_array(n_hat, x)
        for i in range(N):
            lappr[i] *= alpha
            
        success, iterations = dec._decode(lappr, synd, decoder_iterations, final_lappr)          
        
        if (success):
            decoding_iterations += iterations
            successful_decoding += 1
                
        new_errors = count_errors_from_lappr(final_lappr[:K], word[:K])
        if (new_errors):
            frame_error_count += 1
            err_count += new_errors
            
            
        if ((frame_error_count >= ferr_count_min) \
            and (wordcount > simulation_loops/20) ):
            break

    wordcount += 1

    return (snr_dB,
            err_count / (wordcount*K),
            frame_error_count / wordcount,
            0 if (successful_decoding == 0) else decoding_iterations / successful_decoding)




cpdef tuple simulate_direct_snr_dB(double snr_dB, Decoder dec, Matrix mat,
                                   PAMAlphabet pa,
                                   int decoder_iterations, int simulation_loops,
                                   int ferr_count_min):

    cdef long err_count, frame_error_count, decoding_iterations, successful_decoding, N, K, N_symb, wordcount
    cdef double Es, N0, twoVariance, sigma
    cdef long [:] x
    cdef double [:] y
    cdef double [:] lappr
    cdef double [:] final_lappr
    cdef NoiseMapper nm
    cdef unsigned char success
    cdef int iterations
    cdef long new_errors
    
    err_count = 0
    frame_error_count = 0
    
    decoding_iterations = 0
    successful_decoding = 0

    Es = pa.variance
    twoVariance = Es * ( 10**(-snr_dB/10) )
    N0 = twoVariance / 2
    sigma = sqrt(N0)

    nm = NoiseMapper(pa, N0)
    
    N = mat.vnum
    K = N - mat.cnum

    N_symb = N // pa.bit_per_symbol

    lappr       = cvarray(shape=(N,), itemsize=sizeof(double), format="d")
    final_lappr = cvarray(shape=(N,), itemsize=sizeof(double), format="d")
    
    for wordcount in range(simulation_loops):
        
        # Alice generates tx symbols:
        x     = pa.random_symbols(N_symb)
        word  = pa.demap_symbols_to_bits(x)
        synd  = mat.eval_syndrome(word)
        
        # Channel adds AWGN
        y     = np.array(pa.index_to_value(x)) + sigma*np.random.randn(N_symb)
                
        # LAPPR construction on behalf of Bob
        # Opportunistically use final_lappr as buffer
        _y_to_lappr_grey_array(&y[0], N_symb,
                               &pa.constellation[0], pa.bit_per_symbol, pa.order,
                               twoVariance, &lappr[0], &final_lappr[0]) 
        
        # Actually decode
        success, iterations = dec._decode(lappr, synd, decoder_iterations, final_lappr)          
        
        if (success):
            decoding_iterations += iterations
            successful_decoding += 1
            
        new_errors = count_errors_from_lappr(final_lappr[:K], word[:K])
        if (new_errors):
            frame_error_count += 1
            err_count += new_errors
            
            
        if ((frame_error_count >= ferr_count_min) \
            and (wordcount > simulation_loops/20) ):
            print(np.array(y[:2]), " LLR --> ", np.array(lappr[:pa.bit_per_symbol*2]))
            break

    wordcount += 1

    return (snr_dB,
            err_count / (wordcount*K),
            frame_error_count / wordcount,
            0 if (successful_decoding == 0) else decoding_iterations / successful_decoding)



cpdef tuple simulate_hard_reverse_snr_dB(double snr_dB, Decoder dec, Matrix mat,
                                         PAMAlphabet pa,
                                         int decoder_iterations, int simulation_loops,
                                         int ferr_count_min):

    cdef long err_count, frame_error_count, decoding_iterations, successful_decoding, N, K, N_symb, wordcount
    cdef double Es, N0, twoVariance, sigma
    cdef long [:] x
    cdef long [:] x_hat
    cdef double [:] y
    cdef double [:] lappr
    cdef double [:] final_lappr
    cdef NoiseMapper nm
    cdef unsigned char success
    cdef int iterations
    cdef long new_errors
    
    err_count = 0
    frame_error_count = 0
    
    decoding_iterations = 0
    successful_decoding = 0

    Es = pa.variance
    twoVariance = Es * ( 10**(-snr_dB/10) )
    N0 = twoVariance / 2
    sigma = sqrt(N0)

    nm = NoiseMapper(pa, N0)
    
    N = mat.vnum
    K = N - mat.cnum

    N_symb = N // pa.bit_per_symbol

    lappr       = cvarray(shape=(N,), itemsize=sizeof(double), format="d")
    final_lappr = cvarray(shape=(N,), itemsize=sizeof(double), format="d")
    
    for wordcount in range(simulation_loops):
        
        # Alice generates tx symbols:
        x     = pa.random_symbols(N_symb)
        
        # Channel adds AWGN
        y     = np.array(pa.index_to_value(x)) + sigma*np.random.randn(N_symb)

        # Take a decision
        x_hat = nm.hard_decide_index(y)
        word  = pa.demap_symbols_to_bits(x_hat)
        synd  = mat.eval_syndrome(word)
                
        # LLR contruction at Alice's side
        lappr = nm.bare_llr(x)
        
        # Actually decode
        success, iterations = dec._decode(lappr, synd, decoder_iterations, final_lappr)          
        
        if (success):
            decoding_iterations += iterations
            successful_decoding += 1
            
        new_errors = count_errors_from_lappr(final_lappr[:K], word[:K])
        if (new_errors):
            frame_error_count += 1
            err_count += new_errors
            
            
        if ((frame_error_count >= ferr_count_min) \
            and (wordcount > simulation_loops/20) ):
            break

    wordcount += 1

    return (snr_dB,
            err_count / (wordcount*K),
            frame_error_count / wordcount,
            0 if (successful_decoding == 0) else decoding_iterations / successful_decoding)
