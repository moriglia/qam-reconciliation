if __name__=="__main__":
    from qamreconciliation import Matrix as qamMatrix, Decoder as qamDecoder
    from galois import GF2
    import pandas as pd
    import numpy as np
    from parfor import parfor
    
    raw_ber = [1e-5, 1e-4, 1e-3, 0.01, 0.02, 0.03, 0.04, 0.05]
    df = pd.read_csv("sims/dvbs2ldpc0.750.csv")
    niters = 5000
    
    @parfor(raw_ber)
    def final_ber(rber):
        dec = qamDecoder(df.vid[1:].to_numpy(),
                         df.cid[1:].to_numpy())
        mat = qamMatrix(df.vid[1:].to_numpy(),
                        df.cid[1:].to_numpy())
        
        
        error_count = 0
        frame_error_count = 0
        successful_decoding = 0
        decoding_iterations = 0
        
        for it in range(1, niters+1):
            word = GF2.Random(dec.vnum)
            synd = mat.eval_syndrome(word)

            llr = (np.log2(1-rber) - np.log2(rber)) * \
                (1-2*np.array(word + \
                              GF2((np.random.rand(word.size) < rber).astype(np.ubyte)),
                              dtype=np.longdouble))
            
            (success, itcount, lappr_final) = dec.decode(llr, synd, 30)
            
            
            new_errors = np.array(word + GF2((np.array(lappr_final) < 0).astype(np.ubyte)), dtype=int).sum()
            if (new_errors>0):
                error_count += new_errors
                frame_error_count+=1
            if (success):
                successful_decoding += 1
                decoding_iterations += itcount


            if (error_count > 100 and it > 100):
                break
            
        return (rber,
                error_count / (it*dec.vnum),
                frame_error_count / it,
                decoding_iterations/successful_decoding if (successful_decoding > 0) else 0)
    
    
    df = pd.DataFrame(final_ber,
                      columns=["f", "ber", "fer", "iters"])
    
    try:
        df.to_csv("sims/res_dvbs2ldpc0.750_cython.csv")
    except Exception:
        df.to_csv("res_dvbs2ldpc0.750_cython.csv")
