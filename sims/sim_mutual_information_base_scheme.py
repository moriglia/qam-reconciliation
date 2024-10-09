

if __name__=="__main__":
    import argparse
    from qamreconciliation.mutual_information \
        import mutual_information_base_scheme, mutual_information_X_Xhat, \
        mutual_information_X_Y, P_xhat
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
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--gnuplot", action="store_true")

    args = parser.parse_args()




    EsN0dB = np.linspace(args.snr[0], args.snr[1], args.nsnr)


    @parfor(EsN0dB)
    def mutual_information_value(esn0db):
        pa = PAMAlphabet(args.bps, 2)

        Es = pa.variance
        N0 = Es * (10**(-esn0db/10)) / 2

        nm = NoiseMapper(pa, N0)
        p_Xhat = P_xhat(nm)

        I_base_scheme = mutual_information_base_scheme(nm, p_Xhat)
        I_X_Xhat = mutual_information_X_Xhat(nm, p_Xhat)
        I_X_Y = mutual_information_X_Y(nm)

        ebn0db_base  = esn0db - 10*np.log10(I_base_scheme)
        ebn0db_XXhat = esn0db - 10*np.log10(I_X_Xhat)
        ebn0db_XY    = esn0db - 10*np.log10(I_X_Y)
        
        return (esn0db,
                ebn0db_base,
                I_base_scheme,
                ebn0db_XXhat,
                I_X_Xhat,
                ebn0db_XY,
                I_X_Y)



    
    from pandas import DataFrame
    df = DataFrame(mutual_information_value,
                   columns=["EsN0dB",
                            "EbN0dB base",
                            "I(N,X;Xhat)",
                            "EbN0dB X;Xhat",
                            "I(X;Xhat)",
                            "EbN0dB X;Y",
                            "I(X;Y)"])
    df.to_csv(args.out)
    
        
    if args.gnuplot:
        gnuplot_script = f"""
        set datafile separator ","
        set xlabel "E_b/N_0 [dB]"
        set ylabel "I(X, N ; \hat{{X}}) [bit/c.u.]"
        set grid
        
        plot '{args.out}' using 3:4 with lines title "I(X,N;Xhat)", \\
             '{args.out}' using 5:6 with lines title "I(X;Xhat)", \\
             '{args.out}' using 7:8 with lines title "I(X;Y)"
             
        """

        with open(f"{args.out}.gnuplot", 'w') as f:
            f.write(gnuplot_script)
            
    if (args.display):
        from matplotlib import pyplot as plt
        
        plt.plot(df["EbN0dB base"],
                 df["I(N,X;Xhat)"],
                 label="$I(\hat{X} \; ; \; X,\; N)$")
        plt.plot(df["EbN0dB X;Xhat"],
                 df["I(X;Xhat)"],
                 label="$I(X;\hat{X})$")
        plt.plot(df["EbN0dB X;Y"],
                 df["I(X;Y)"],
                 label="$I(X;Y)$")
        plt.xlabel("$E_b/N_0$ [dB]")
        plt.grid("both")
        plt.legend()
        plt.show()
