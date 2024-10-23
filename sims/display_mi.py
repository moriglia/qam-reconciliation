# SPDX-License-Identifier: GPL-3.0-or-later
#     Copyright (C) 2024  Marco Origlia
# 
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
# 
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
# 
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

if __name__=="__main__":
    import argparse
    

    parser = argparse.ArgumentParser(
        prog="display_mi",
        description="Display mutual information file"
    )

    
    parser.add_argument("file")
    parser.add_argument("--title", default="--title [TITLE]")
    parser.add_argument("--rescalex", action="store_true")
    parser.add_argument("--extra-file", type=str, required=False)
    parser.add_argument("--extra-file-label", type=str, required=False, default="extra file")
    args = parser.parse_args()

    import pandas as pd

    df = pd.read_csv(args.file)
    
    from matplotlib import pyplot as plt


    extra_matlab_keys = ["I_HD_MATLAB", "I_X_Y_MATLAB"]

    if (args.rescalex):
        import numpy as np
        plt.plot(df["EsN0dB"] - 10*np.log10(df["I(N,X;Xhat)"]),
                 df["I(N,X;Xhat)"],
                 label="$I(\hat{X} \; ; \; X,\; N)$")
        plt.plot(df["EsN0dB"] - 10*np.log10(df["I(X;Xhat)"]),
                 df["I(X;Xhat)"],
                 label="$I(X;\hat{X})$")
        plt.plot(df["EsN0dB"] - 10*np.log10(df["I(X;Y)"]),
                 df["I(X;Y)"],
                 label="$I(X;Y)$")
        plt.xlabel("$E_b/N_0$ [dB]")

        if (args.extra_file):
            dfextra = pd.read_csv(args.extra_file)
            plt.plot(dfextra["EsN0dB"] - 10*np.log10(dfextra["I(N,X;Xhat)"]),
                     dfextra["I(N,X;Xhat)"],
                     label=f"$I(\hat{{X}} \; ; \; X,\; N)$ {args.extra_file_label}",
                     linestyle="--")
            plt.plot(dfextra["EsN0dB"] - 10*np.log10(dfextra["I(X;Xhat)"]),
                     dfextra["I(X;Xhat)"],
                     label=f"$I(X;\hat{{X}})$ {args.extra_file_label}",
                     linestyle="--")
            plt.plot(dfextra["EsN0dB"] - 10*np.log10(dfextra["I(X;Y)"]),
                     dfextra["I(X;Y)"],
                     label=f"$I(X;Y)$ {args.extra_file_label}",
                     linestyle="--")
            

        for key in extra_matlab_keys:
            if (key in df.keys()):
                if ("SNR_MATLAB" in df.keys()):
                    plt.plot(df["SNR_MATLAB"] - 10*np.log10(df[key]),
                             df[key],
                             label=key,
                             linestyle=":")
                else:
                    plt.plot(df["EsN0dB"] - 10*np.log10(df[key]),
                             df[key],
                             label=key,
                             linestyle=":")

    else:
        plt.plot(df["EsN0dB"],
                 df["I(N,X;Xhat)"],
                 label="$I(\hat{X} \; ; \; X,\; N)$")
        plt.plot(df["EsN0dB"],
                 df["I(X;Xhat)"],
                 label="$I(X;\hat{X})$")
        plt.plot(df["EsN0dB"],
                 df["I(X;Y)"],
                 label="$I(X;Y)$")
        plt.xlabel("$E_s/N_0$ [dB]")

        if (args.extra_file):
            dfextra = pd.read_csv(args.extra_file)
            plt.plot(dfextra["EsN0dB"],
                     dfextra["I(N,X;Xhat)"],
                     label=f"$I(\hat{{X}} \; ; \; X,\; N)$ {args.extra_file_label}",
                     linestyle="--")
            plt.plot(dfextra["EsN0dB"],
                     dfextra["I(X;Xhat)"],
                     label=f"$I(X;\hat{{X}})$ {args.extra_file_label}",
                     linestyle="--")
            plt.plot(dfextra["EsN0dB"],
                     dfextra["I(X;Y)"],
                     label=f"$I(X;Y)$ {args.extra_file_label}",
                     linestyle="--")

        for key in extra_matlab_keys:
            if (key in df.keys()):
                if ("SNR_MATLAB" in df.keys()):
                    plt.plot(df["SNR_MATLAB"],
                             df[key],
                             label=key,
                             linestyle=":")
                else:
                    plt.plot(df["EsN0dB"],
                             df[key],
                             label=key,
                             linestyle=":")    
    plt.grid("both")
    plt.legend()
    plt.title(args.title)
    plt.ylabel("Mutual information bits/c.u.")
    plt.show()
