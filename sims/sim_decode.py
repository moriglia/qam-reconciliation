# -*- mode: python ; gud -*-

if __name__=="__main__":
    import argparse
    from parfor import parfor
    from galois import GF2
    import qamreconciliation as qamr
    import numpy as np
    import pandas as pd
    
    
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
    

    args = parser.parse_args()

    
    edge_df = pd.read_csv(args.edgefile)
    mat = qamr.Matrix(edge_df, args.first_row)
    N = mat.vnum

    # input_error_count = [N//20,  N//25,   N//30,   N//40,   N//50,  N//60,
    #                      N//75,  N//100,  N//120,  N//150,  N//160, N//200,
    #                      N//600, N//1000, N//2000, N//5000, 5,      1]

    # input_error_count = [N//10, N//20, N//25, N//30, N//35, N//40, N//45, N//50, N//100]

    EbN0dB = np.linspace(0, 5, 11)
    
    @parfor(EbN0dB)
    def final_ber(EbN0dB_val):

        sigma = np.sqrt(10**(-EbN0dB_val/10))
        
        err_count = 0
        frame_error_count = 0

        decoding_iterations = 0
        successful_decoding = 0

        dec = qamr.Decoder(edge_df)
        
        for wordcount in range(args.simloops):
            # error_vector = GF2([*[1]*num_errors, *[0]*(N-num_errors)])
            # np.random.shuffle(error_vector)
            
            word = GF2.Random(N)
            synd = mat.eval_syndrome(word)

            # rcvs = -2*np.array(word, dtype=int)+1 + sigma*np.random.randn(word.size)
            # llr = 2*rcvs/sigma**2 * np.log2(np.exp(1))
            llr = 2*np.log2(np.exp(1))/sigma**2 * (-2*np.array(word, dtype=int)+1 + \
                                                   sigma*np.random.randn(word.size))
            
            (success, itcount, lappr_final) = dec.decode(llr, synd, args.maxiter)
            
            if (success):
                decoding_iterations += itcount
                successful_decoding += 1
                continue

            err_count += np.array(GF2((np.array(lappr_final) < 0).astype(int)) + word,
                                  dtype=int).sum()
            frame_error_count += 1
            
            if (err_count >= args.minerr and wordcount > args.simloops):
                break


        wordcount = wordcount + 1
        return (EbN0dB_val,
                err_count           / (wordcount*N),
                frame_error_count   / wordcount,
                0 if (successful_decoding == 0) else decoding_iterations / successful_decoding)


    df = pd.DataFrame(final_ber,
                      columns=['EbN0dB',
                               'ber',
                               'fer',
                               'iters'])

    df.to_csv(args.out)
    
