
if __name__=="__main__":
    import argparse
    

    parser = argparse.ArgumentParser(
        prog="display_mi",
        description="Display mutual information file"
    )

    
    parser.add_argument("file")
    parser.add_argument("--title", default="--title [TITLE]")
    parser.add_argument("--rescalex", action="store_true")
    parser.add_argument("--logy", action="store_true")
    parser.add_argument("--reference-file", type=str, required=False)
    parser.add_argument("--extra-file", type=str, required=False)
    args = parser.parse_args()

    import pandas as pd

    df = pd.read_csv(args.file)
    
    from matplotlib import pyplot as plt
    if (args.logy):
        fun = plt.semilogy
    else:
        fun = plt.plot
    extra_plot_args = {}


    extra_matlab_keys = ["I_HD_MATLAB", "I_X_Y_MATLAB"]

    if (args.rescalex):
        import numpy as np
        for key in df.keys()[2:]:
            fun(df["EsN0dB"] - 10*np.log10(df[key]),
                df[key],
                label=key,
                **extra_plot_args)
        if (args.extra_file):
            dfextra = pd.read_csv(args.extra_file)
            for key in dfextra.keys()[2:]:
                fun(dfextra["EsN0dB"] - 10*np.log10(dfextra[key]),
                    dfextra[key],
                    label=f"{key} extra",
                    **extra_plot_args)
                
        plt.xlabel("$E_b/N_0$ [dB]", fontsize=18)
        if (args.reference_file):
            dfref = pd.read_csv(args.reference_file)
            fun(dfref["EsN0dB"] - 10*np.log10(dfref["I(X;Y)"]),
                dfref["I(X;Y)"],
                label="I(X;Y)",
                linestyle=":",
                **extra_plot_args)
            
            fun(dfref["EsN0dB"] - 10*np.log10(dfref["I(X;Xhat)"]),
                dfref["I(X;Xhat)"],
                label="I(X;Xhat)",
                linestyle="-.",
                **extra_plot_args)
            

    else:
        for key in df.keys()[2:]:
            fun(df["EsN0dB"],
                df[key],
                label=key,
                **extra_plot_args)

        if (args.extra_file):
            dfextra = pd.read_csv(args.extra_file)
            for key in dfextra.keys()[2:]:
                fun(dfextra["EsN0dB"],
                    dfextra[key],
                    label=f"{key} extra",
                    **extra_plot_args)

        if (args.reference_file):
            dfref = pd.read_csv(args.reference_file)
            fun(dfref["EsN0dB"],
                dfref["I(X;Y)"],
                label="I(X;Y)",
                linestyle=":",
                **extra_plot_args)

            fun(dfref["EsN0dB"],
                dfref["I(X;Xhat)"],
                label="I(X;Xhat)",
                linestyle=":",
                **extra_plot_args)


        plt.xlabel("$E_s/N_0$ [dB]", fontsize=18)

    prop = {'size' : 18}
    plt.grid("both")
    plt.legend(prop=prop)
    plt.title(args.title, fontsize=22)
    plt.ylabel("Mutual information bits/c.u.", fontsize=18)
    plt.show()
