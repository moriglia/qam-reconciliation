from cython.parallel import prange, threadid, parallel
from qamreconciliation.alphabet cimport PAMAlphabet
from qamreconciliation.matrix cimport Matrix
from qamreconciliation cimport Decoder, NoiseMapper, NoiseMapperFlipSign
from cython.view import array as cvarray
from libc.math import sqrt, randn
from qamreconciliation.decoder cimport DecodingResult
from qamreconciliation.utils cimport count_errors_from_lappr


cpdef double [:] ber_loop(double EsN0dB_val,
                          PAMAlphabet pa,
                          Matrix pcm,
                          Decoder [:] decoder_pool,
                          int decoder_iter,
                          int simloops,
                          int minerr,
                          int pool_size):
    cdef double [:] res = cvarray(shape=(4,), itemsize=sizeof(double), format="d")
    cdef int wordcount, chunkcount
    # Modulation and Channel parameters
    cdef float Es, N0, sigma

    # Code parameters
    cdef long N, K
    cdef long N_symb

    # Extra variables
    cdef long i

    # channel and side-information variables
    cdef:
        long [:] x
        double [:] y
        long [:] x_hat
        double [:] n_hat
        unsigned char [:] word
        unsigned char [:] synd
        double [:] lappr

    # decoding results
    cdef:
        DecodingResult dec_res
        # unsigned char success
        # long itcount
        # double [:] lappr_final
    # 
    cdef:
        long err_count = 0
        long decoding_iterations = 0
        long successful_decoding = 0
        long new_errors
        long frame_error_count = 0

    Es = pa.variance
    N0 = Es * (10**(-EsN0dB_val/10))/2
    sigma = sqrt(N0)
    
    
    noiseMapper = NoiseMapperFlipSign(pa, N0)
        
    N = pcm.vnode_num
    K = N - pcm.cnode_num
    N_symb = N//pa.bit_per_symbol
    
    for chunkcount in range(10):
        with nogil:
            for wordcount in prange(simloops//10,
                                    num_threads=pool_size):

                with gil:
                    # Alice generates tx symbols:
                    x = pa.random_symbols(N_symb)
                    
                    # Channel adds AWGN
                    y = pa.idex_to_value(x)
                    
                for i in range(N_symb):
                    y[i] += sigma*randn()

                
                # Bob decides for the symbols based on the thresholds
                x_hat = noiseMapper.hard_decide_index(y)
                n_hat = noiseMapper.map_noise(y, x_hat)
                
                # Bob evaluates the received bits and the evaluates the syndrome
                word = pa.demap_symbols_to_bits(x_hat)
                synd = pcm.eval_syndrome(word)
                
                # Alice uses the transformed noise and the syndrome received from Bob
                # to build the LLR and proceed with reconciliation
                lappr = noiseMapper.demap_lappr_array(n_hat, x)
                
                dec_res = decoder_pool[threadid()].decode(lappr, synd, decoder_iter)            
            
                if (dec_res.success):
                    decoding_iterations += dec_res.iter_number
                    successful_decoding += 1
                
                new_errors = count_errors_from_lappr(dec_res.llr[:K], word[:K])
                if (new_errors):
                    frame_error_count += 1
                    err_count += new_errors
                    
                    
        if (err_count >= minerr):
            break

    
    res[0] = EsN0dB_val
    res[1] = err_count / (chunkcount * (simloops//10) * K)
    res[2] = frame_error_count / (chunkcount * (simloops//10))
    res[3] = 0 if (successful_decoding == 0) else decoding_iterations / successful_decoding

    return res
    
