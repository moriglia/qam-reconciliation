# -*- mode: python ; gud -*-

if __name__=="__main__":
    import argparse
    from parfor import parfor
    import numpy as np
    import pandas as pd
    from qamreconciliation import Decoder, NoiseMapper, NoiseMapperFlipSign, NoiseMapperAntiFlipSign, Matrix
    from qamreconciliation.alphabet import PAMAlphabet
    from qamreconciliation.utils import count_errors_from_lappr
    
    
    parser = argparse.ArgumentParser(
        prog="sim_decode",
        description="Evaluate BER for LDPC codes vs Raw BER"
    )

    parser.add_argument("edgefile")
    parser.add_argument("--out", default="out.csv")

    parser.add_argument("--maxiter",      default = 30, type=int)
    parser.add_argument("--minerr",       default = 20, type=int)
    parser.add_argument("--simloops",     default = 30, type=int)
    parser.add_argument("--snr", type=float, nargs=2, default=[0,5])
    parser.add_argument("--nsnr", type=int, default=11)
    parser.add_argument("--bps", type=int, default=2)

    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--flipsign", action="store_true")
    parser.add_argument("--antiflipsign", action="store_true")
    parser.add_argument("--configuration-base", action="store_true")
    
    args = parser.parse_args()

    
    edge_df = pd.read_csv(args.edgefile)
    N_symb = edge_df.vid[0] // args.bps

    EsN0dB = np.linspace(args.snr[0], args.snr[1], args.nsnr)
    
    @parfor(EsN0dB)
    def final_ber(EsN0dB_val):
        
        err_count = 0
        frame_error_count = 0

        decoding_iterations = 0
        successful_decoding = 0

        dec = Decoder(edge_df.vid[1:].to_numpy(),
                      edge_df.cid[1:].to_numpy())
        mat = Matrix(edge_df.vid[1:].to_numpy(),
                     edge_df.cid[1:].to_numpy())
        pa = PAMAlphabet(args.bps, 2)

        Es = pa.variance
        N0 = Es * (10**(-EsN0dB_val/10))/2
        sigma = np.sqrt(N0)

        if (args.antiflipsign):
            noiseMapper = NoiseMapperAntiFlipSign(pa, N0)
        elif (args.flipsign):
            noiseMapper = NoiseMapperFlipSign(pa, N0)
        elif (args.configuration_base):
            noiseMapper = NoiseMapper(pa, N0)
        else:
            config = np.zeros(pa.order, dtype=np.uint8)
            config[1::2] = 1
            noiseMapper = NoiseMapper(pa, N0, config)
        
        N = edge_df.vid[0]
        K = N - edge_df.cid[0]
        
        for wordcount in range(args.simloops):

            # Alice generates tx symbols:
            x = pa.random_symbols(N_symb)

            # Channel adds AWGN
            y = np.array(pa.index_to_value(x)) + sigma*np.random.randn(N_symb)

            # Bob decides for the symbols based on the thresholds
            x_hat = noiseMapper.hard_decide_index(y)
            n_hat = noiseMapper.map_noise(y, x_hat)

            # Bob evaluates the received bits and the evaluates the syndrome
            word = pa.demap_symbols_to_bits(x_hat)
            synd = mat.eval_syndrome(word)

            # Alice uses the transformed noise and the syndrome received from Bob
            # to build the LLR and proceed with reconciliation
            lappr = noiseMapper.demap_lappr_array(n_hat, x)
            
            (success, itcount, lappr_final) = dec.decode(lappr, synd, args.maxiter)            
            
            if (success):
                decoding_iterations += itcount
                successful_decoding += 1
                
            new_errors = count_errors_from_lappr(lappr_final[:K], word[:K])
            if (new_errors):
                frame_error_count += 1
                err_count += new_errors

                            
            if (frame_error_count >= args.minerr \
                and wordcount > args.simloops/20 ):
                break


        wordcount += 1
        return (EsN0dB_val,
                err_count           / (wordcount*K),
                frame_error_count   / wordcount,
                0 if (successful_decoding == 0) else decoding_iterations / successful_decoding # ,
                # err_count_bare           / (wordcount*K),
                # frame_error_count_bare   / wordcount,
                # 0 if (successful_decoding_bare == 0) else decoding_iterations_bare / successful_decoding_bare
                )


    df = pd.DataFrame(final_ber,
                      columns=['EsN0dB',
                               'ber',
                               'fer',
                               'iters'])

    df.to_csv(args.out)
    
