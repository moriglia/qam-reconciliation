

if __name__=="__main__":
    import argparse
    from qamreconciliation.mutual_information \
         import P_xhat, montecarlo_information
    from qamreconciliation.alphabet import PAMAlphabet
    from qamreconciliation.noisemapper import NoiseMapper
    import numpy as np
    from parfor import parfor
    

    parser = argparse.ArgumentParser(
        prog="mutual_information_base_scheme",
        description="Evaluate mutual information vs SNR of the base scheme"
    )

    parser.add_argument("--out", default="out.csv")
    parser.add_argument("--snr", type=float, nargs=2, default=[-20,20])
    parser.add_argument("--nsnr", type=int, default=401)
    parser.add_argument("--bps", type=int, default=2)
    parser.add_argument("--niters", type=int, default=(1<<8))
    parser.add_argument("--samples-per-iter", type=int, default = (1<<12))
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

        I_X_Xhat = 0
        I_X_Y = 0
        I_XN_Xhat = 0

        for i in range(args.niters):
            I_X_Xhat_tmp, I_X_Y_tmp, I_XN_Xhat_tmp = montecarlo_information(
                pa, nm, p_Xhat, args.samples_per_iter
            )

            I_X_Xhat  += I_X_Xhat_tmp
            I_X_Y     += I_X_Y_tmp
            I_XN_Xhat += I_XN_Xhat_tmp

        I_X_Xhat  /= args.niters
        I_X_Y     /= args.niters
        I_XN_Xhat /= args.niters
        
        return (esn0db,
                I_X_Xhat,
                I_X_Y,
                I_XN_Xhat)



    
    from pandas import DataFrame
    df = DataFrame(mutual_information_value,
                   columns=["EsN0dB",
                            "I(X;Xhat)",
                            "I(X;Y)",
                            "I(N,X;Xhat)"])
    df.to_csv(args.out)
    
        
    if args.gnuplot:
        gnuplot_script = f"""
        set datafile separator ","
        set xlabel "E_b/N_0 [dB]"
        set ylabel "I(X, N ; \hat{{X}}) [bit/c.u.]"
        set grid
        
        plot '{args.out}' using 2:5 with lines title "I(X,N;Xhat)", \\
             '{args.out}' using 2:3 with lines title "I(X;Xhat)", \\
             '{args.out}' using 2:4 with lines title "I(X;Y)"
             
        """

        with open(f"{args.out}.gnuplot", 'w') as f:
            f.write(gnuplot_script)
            
    if (args.display):
        from matplotlib import pyplot as plt
        
        plt.plot(df["EsN0dB"],
                 df["I(N,X;Xhat)"],
                 label="$I(\hat{X} \; ; \; X,\; N)$")
        plt.plot(df["EsN0dB"],
                 df["I(X;Xhat)"],
                 label="$I(X;\hat{X})$")
        plt.plot(df["EsN0dB"],
                 df["I(X;Y)"],
                 label="$I(X;Y)$")
        plt.xlabel("$E_b/N_0$ [dB]")
        plt.grid("both")
        plt.legend()
        plt.show()
