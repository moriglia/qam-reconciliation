# -*- mode: python ; gud -*-

if __name__=="__main__":
    import argparse
    from parfor import parfor
    from galois import GF2
    import qamreconciliation as qamr
    import numpy as np
    import pandas as pd
    import scipy as sp
    from qamreconciliation.decoder import Decoder as CyDecoder
    
    parser = argparse.ArgumentParser(
        prog="sim_decode",
        description="Evaluate BER for LDPC codes vs Raw BER"
    )

    parser.add_argument("edgefile")
    # parser.add_argument("codewordfile")
    parser.add_argument("--out", default="out.csv")

    parser.add_argument("--maxiter",      default = 30, type=int)
    parser.add_argument("--minerr",       default = 20, type=int)
    parser.add_argument("--first_row",    default = True,
                        action="store_true",
                        help="Flag: does the first line of the csv contain the number of edges")
    parser.add_argument("--simloops",     default = 30, type=int)
    parser.add_argument("--snr", type=float, nargs=2, default=[0,5])
    parser.add_argument("--nsnr", type=int, default=11)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--hard", action='store_true', default=False)

    args = parser.parse_args()

    
    edge_df = pd.read_csv(args.edgefile)
    
    EbN0dB = np.linspace(args.snr[0], args.snr[1], args.nsnr)
    
    @parfor(EbN0dB)
    def final_ber(EbN0dB_val):

        v = (10**(-EbN0dB_val/10))/2
        vsqrt = np.sqrt(v)
        
        err_count = 0
        frame_error_count = 0

        decoding_iterations = 0
        successful_decoding = 0

        dec = CyDecoder(edge_df.vid[1:].to_numpy(),
                        edge_df.cid[1:].to_numpy())
        mat = qamr.Matrix(edge_df.vid[1:].to_numpy(),
                          edge_df.cid[1:].to_numpy())
        N = mat.vnum
        K = N - mat.cnum


        if args.hard:
            err_prob = 0.5*sp.special.erfc(1/(np.sqrt(2)*vsqrt))
            LLR0 = np.log((1-err_prob)/err_prob)
            
            for wordcount in range(args.simloops):
                
                word = GF2.Random(N)
                synd = mat.eval_syndrome(word)
                
                llr = LLR0 * \
                    np.sign((1 - 2.0*np.array(word, dtype=np.double) + \
                             vsqrt*np.random.randn(word.size)))
                
                (success, itcount, lappr_final) = dec.decode(llr, synd, args.maxiter)
                
                if (success):
                    decoding_iterations += itcount
                    successful_decoding += 1
                    # continue
                    
                new_errors = np.array(GF2((np.array(lappr_final[:K]) < 0).astype(np.ubyte)) + word[:K],
                                      dtype=int).sum()
                if (new_errors):
                    frame_error_count += 1
                    err_count += new_errors
                    
                if (err_count >= args.minerr and wordcount > args.simloops/20):
                    # print(f"[{EbN0dB_val}, iteration {wordcount}] found {err_count} errors")
                    break
        else:
            for wordcount in range(args.simloops):
                
                word = GF2.Random(N)
                synd = mat.eval_syndrome(word)
                
                # rcvs = -2*np.array(word, dtype=int)+1 + sigma*np.random.randn(word.size)
                # llr = 2*rcvs/sigma**2 * np.log2(np.e)
                # llr = 2*np.log2(np.e)/(sigma**2) * \
                llr = 2*args.alpha/v * \
                    (-2*np.array(word, dtype=np.double)+1 + \
                     vsqrt*np.random.randn(word.size))
                
                (success, itcount, lappr_final) = dec.decode(llr, synd, args.maxiter)
                
                if (success):
                    decoding_iterations += itcount
                    successful_decoding += 1
                    # continue
                    
                new_errors = np.array(GF2((np.array(lappr_final[:K]) < 0).astype(np.ubyte)) + word[:K],
                                      dtype=int).sum()
                if (new_errors):
                    frame_error_count += 1
                    err_count += new_errors
                    
                if (err_count >= args.minerr and wordcount > args.simloops/20):
                    # print(f"[{EbN0dB_val}, iteration {wordcount}] found {err_count} errors")
                    break


        wordcount += 1
        return (EbN0dB_val,
                err_count           / (wordcount*K),
                frame_error_count   / wordcount,
                0 if (successful_decoding == 0) else decoding_iterations / successful_decoding)


    df = pd.DataFrame(final_ber,
                      columns=['EbN0dB',
                               'ber',
                               'fer',
                               'iters'])

    df.to_csv(args.out)
    
