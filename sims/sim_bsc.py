if __name__=="__main__":
    from qamreconciliation import Matrix as qamMatrix
    from qamreconciliation.decoder_cy import Decoder as qamDecoder
    from galois import GF2
    import pandas as pd
    import numpy as np
    from parfor import parfor
    import argparse

    parser = argparse.ArgumentParser(
        prog="sim_bsc",
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
    parser.add_argument("--rber", type=float, nargs=2, default=[0.01,0.04])
    parser.add_argument("--rpoints", type=int, default=31)

    args = parser.parse_args()

    
    # raw_ber = [1e-5, 1e-4, 1e-3, 0.01, 0.02, 0.03, 0.04, 0.05]
    raw_ber = np.linspace(args.rber[0], args.rber[1], args.rpoints)
    df = pd.read_csv(args.edgefile)
    niters = args.simloops
    
    @parfor(raw_ber)
    def final_ber(rber):
        if (args.first_row):
            dec = qamDecoder(df.vid[1:].to_numpy(),
                             df.cid[1:].to_numpy())
            mat = qamMatrix(df.vid[1:].to_numpy(),
                            df.cid[1:].to_numpy())
        else:
            dec = qamDecoder(df.vid.to_numpy(),
                             df.cid.to_numpy())
            mat = qamMatrix(df.vid.to_numpy(),
                            df.cid.to_numpy())
        
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
            
            (success, itcount, lappr_final) = dec.decode(llr, synd, args.maxiter)
            
            
            new_errors = np.array(word + GF2((np.array(lappr_final) < 0).astype(np.ubyte)), dtype=int).sum()
            if (new_errors>0):
                error_count += new_errors
                frame_error_count+=1
            if (success):
                successful_decoding += 1
                decoding_iterations += itcount


            if (error_count > args.minerr and it > max(20, niters//100)):
                break

        print(f"[RawBER={rber}] it={it}, error_count={error_count}, dit={decoding_iterations}, succ={successful_decoding}")
        return (rber,
                error_count / (it*dec.vnum),
                frame_error_count / it,
                decoding_iterations/successful_decoding if (successful_decoding > 0) else 0)
    
    
    df = pd.DataFrame(final_ber,
                      columns=["f", "ber", "fer", "iters"])
    
    try:
        df.to_csv(args.out)
    except Exception:
        df.to_csv("out.csv")
