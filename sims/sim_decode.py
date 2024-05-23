# -*- mode: python ; gud -*-

if __name__=="__main__":
    import argparse
    from parfor import parfor
    from galois import GF2
    import qamreconciliation as qamr
    import numpy as np
    import pandas as pd
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

    args = parser.parse_args()

    
    edge_df = pd.read_csv(args.edgefile)
    
    # input_error_count = [N//20,  N//25,   N//30,   N//40,   N//50,  N//60,
    #                      N//75,  N//100,  N//120,  N//150,  N//160, N//200,
    #                      N//600, N//1000, N//2000, N//5000, 5,      1]

    # input_error_count = [N//10, N//20, N//25, N//30, N//35, N//40, N//45, N//50, N//100]

    EbN0dB = np.linspace(args.snr[0], args.snr[1], args.nsnr)
    
    @parfor(EbN0dB)
    def final_ber(EbN0dB_val):

        sigma = np.sqrt(10**(-EbN0dB_val/10))
        
        err_count = 0
        frame_error_count = 0

        decoding_iterations = 0
        successful_decoding = 0

        dec = CyDecoder(edge_df.vid[1:].to_numpy(),
                        edge_df.cid[1:].to_numpy())
        mat = qamr.Matrix(edge_df.vid[1:].to_numpy(),
                          edge_df.cid[1:].to_numpy())
        N = mat.vnum
    
        for wordcount in range(args.simloops):
        
            word = GF2.Random(N)
            synd = mat.eval_syndrome(word)

            # rcvs = -2*np.array(word, dtype=int)+1 + sigma*np.random.randn(word.size)
            # llr = 2*rcvs/sigma**2 * np.log2(np.e)
            # llr = 2*np.log2(np.e)/(sigma**2) * \
            llr = 2*args.alpha/(sigma**2) * \
                (-2*np.array(word, dtype=np.double)+1 + \
                 sigma*np.random.randn(word.size))
            
            (success, itcount, lappr_final) = dec.decode(llr, synd, args.maxiter)
            
            if (success):
                decoding_iterations += itcount
                successful_decoding += 1
                # continue

            new_errors = np.array(GF2((np.array(lappr_final[:N//2]) < 0).astype(np.ubyte)) + word[:N//2],
                                  dtype=int).sum()
            if (new_errors):
                frame_error_count += 1
                err_count += new_errors
        
            if (err_count >= args.minerr and wordcount > args.simloops/20):
                print(f"[{EbN0dB_val}, iteration {wordcount}] found {err_count} errors")
                break


        wordcount += 1
        return (EbN0dB_val,
                err_count           / (wordcount*(N//2)),
                frame_error_count   / wordcount,
                0 if (successful_decoding == 0) else decoding_iterations / successful_decoding)


    df = pd.DataFrame(final_ber,
                      columns=['EbN0dB',
                               'ber',
                               'fer',
                               'iters'])

    df.to_csv(args.out)
    
