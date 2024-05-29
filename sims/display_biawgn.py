from matplotlib import pyplot as plt
import pandas as pd
import scipy as sp
import numpy as np



def h2(p: float):
    if (p < 0 or p > 1):
        raise ValueError(f"probability {p} out of range [0, 1]")
    if ((p == 1) or (p ==  0)):
        return 0
    return -(p*np.log2(p) + (1-p)*np.log2(1-p))


def Cbiawgn_symb(snr: float):
    sqsnr = np.sqrt(snr)
    expsnr = np.exp(-snr)
    invsqpi = 1/np.sqrt(np.pi)
    invlog2 = 1/np.log(2)
    return 1 - 2*sqsnr*invlog2 * \
        (expsnr*invsqpi - sqsnr*sp.special.erfc(sqsnr)) \
        - expsnr / (1 + 2*sqsnr*invsqpi*invlog2)


def phi_root_locus(p_b, snr, R):
    return h2(p_b) - 1 + Cbiawgn_symb(snr)/R


df = pd.read_csv("res_dvbs2ldpc0.500_cython_latest.csv")
# dfalpha09 = pd.read_csv("res_dvbs2ldpc0.500_cython_latest_alpha0.9.csv")
# dfalpha11 = pd.read_csv("res_dvbs2ldpc0.500_cython_latest_alpha1.1.csv")

df_info_only = pd.read_csv("res_dvbs2ldpc0.500_50iter.csv")
df_info_only_hard = pd.read_csv("res_dvbs2ldpc0.500_50iter_hard.csv")

snr_range = np.linspace(-10, 10, 201)
p_acceptable = np.empty(201, dtype=np.float64)

for i in range(len(snr_range)):
    try:
        p_acceptable[i] = sp.optimize.brentq(
            phi_root_locus,
            a=0, b=0.5,
            args=(10**(snr_range[i]/10), 0.5)
        )
    except ValueError as ve:
        print(ve)
        p_acceptable[i] = 0
        
matlab_ber = pd.read_csv("matlab/res_0.500_50iter.csv", header=None).to_numpy()
matlab_ber_hard = pd.read_csv("matlab/res_0.500_50iter_hard.csv", header=None).to_numpy()
# dfpy = pd.read_csv("res_dvbs2ldpc0.500_python.csv")


plt.semilogy(matlab_ber[0], matlab_ber[1], marker="o", markerfacecolor='none', linestyle="--", label="MATLAB 50 iter. soft")
plt.semilogy(matlab_ber_hard[0], matlab_ber_hard[1], marker="o", markerfacecolor='none', linestyle="--", label="MATLAB 50 iter. hard")
plt.semilogy(df.EbN0dB, df.ber, marker="o", label="20 iter. soft")
# plt.semilogy(dfpy.EbN0dB, dfpy.ber, marker="+", label="Python Decoder")
# plt.semilogy(dfalpha09.EbN0dB, dfalpha09.ber, marker="o", label="$alpha=0.9$")
# plt.semilogy(dfalpha11.EbN0dB, dfalpha11.ber, marker="o", label="$alpha=1.1$")
plt.semilogy(snr_range, p_acceptable, linestyle=":", label="Shannon limit")
plt.semilogy(snr_range, 0.5*(1-sp.special.erf(np.sqrt(10**(snr_range/10)/2))), label="No code")
plt.semilogy(df_info_only.EbN0dB+3, df_info_only.ber, marker="x", linestyle="--", label="Ber 50 iter. soft")
plt.semilogy(df_info_only_hard.EbN0dB+3, df_info_only_hard.ber, marker="x", linestyle="--", label="Ber 50 iter. hard")

plt.grid(True, which='both')
plt.legend()
plt.xlabel("$E_s/N_0$ [dB]")
plt.ylabel("Codeword BER $p_b$")
plt.title("BER vs SNR for a 1/2 rate code")

plt.show()
