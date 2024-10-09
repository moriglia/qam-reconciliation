

if __name__=="__main__":
    import argparse
    from qamreconciliation.mutual_information \
        import mutual_information_base_scheme, montecarlo_information, P_xhat
    from qamreconciliation.alphabet import PAMAlphabet
    from qamreconciliation.noisemapper import NoiseMapper
    import numpy as np
    from parfor import parfor
    

    parser = argparse.ArgumentParser(
        prog="mutual_information_base_scheme",
        description="Evaluate mutual information vs SNR of the base scheme"
    )

    parser.add_argument("--out", default="out.csv")
    parser.add_argument("--snr", type=float, nargs=2, default=[0,5])
    parser.add_argument("--nsnr", type=int, default=11)
    parser.add_argument("--bps", type=int, default=2)
    # parser.add_argument("--integration-step-count", type=int, default=1000)
    parser.add_argument("--montecarlo", action="store_true")
    parser.add_argument("--nmontecarlo", type=int, default = 1<<12)
    parser.add_argument("--nloops", type=int, default = 1<<6 )

    args = parser.parse_args()


    M = 1<<args.bps

    
    def reverse_flip_bits(n):
        res = 0
        for k in range(M):
            res += (((n >> k) & 0b1) ^ 0b1) << (M-1 - k)
        return res

    
    def index_to_config(n):
        a = np.empty(M, dtype=np.uint8)
        for i in range(M):
            if ((n>>i) & 0b1):
                a[i] = 1
            else:
                a[i] = 0

        return a

    
    config_list = []
    column_list = ["EsN0dB"]
    for c in range(1<<M):
        if (reverse_flip_bits(c) >= c):
            config_list.append(index_to_config(c))
            column_list.append(f"I(X,N;Xhat)_{c}")
            

    config_array = np.array(config_list)
    config_count = (1<<((M>>1) - 1)) * ((1<<(M>>1))+1)
    print(config_count)
    print(config_array)

    EsN0dB = np.linspace(args.snr[0], args.snr[1], args.nsnr)

    
    @parfor(EsN0dB)
    def mutual_information_value(esn0db):
        pa = PAMAlphabet(args.bps, 2)

        Es = pa.variance
        N0 = Es * (10**(-esn0db/10)) / 2

        if (args.montecarlo):
            res = [esn0db]
            for k in range(config_count):
                nm = NoiseMapper(pa, N0, config_array[k])
                p_Xhat = P_xhat(nm)
                I = 0
                for loopnumber in range(args.nloops):
                    _, _, tmp = montecarlo_information(pa, nm, p_Xhat,
                                                       args.nmontecarlo,
                                                       np.array([0,0,1], dtype=np.uint8))
                    I += tmp
                res.append(I/args.nloops)
            
        else:
            nm = NoiseMapper(pa, N0, config_array[0])
            p_Xhat = P_xhat(nm)
            res = [esn0db, mutual_information_base_scheme(nm, p_Xhat)]
            for k in range(1,config_count):
                nm = NoiseMapper(pa, N0, config_array[k])
                res.append(mutual_information_base_scheme(nm, p_Xhat))
                
        return res



    
    from pandas import DataFrame
    df = DataFrame(mutual_information_value,
                   columns=column_list)
    df.to_csv(args.out)
    
    
