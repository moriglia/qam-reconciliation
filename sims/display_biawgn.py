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


def phi_root_locus(p_b, snr, R):
    return h2(p_b) - 1 + np.log2(1+snr)/R


df = pd.read_csv("res_dvbs2ldpc0.500_cython_latest.csv")



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
        
matlab_ber = pd.read_csv("matlab/res_dvbs2ldpc_0.500_matlab.csv", header=None)
dfpy = pd.read_csv("res_dvbs2ldpc0.500_python.csv")


plt.semilogy(matlab_ber[0], matlab_ber[1], linestyle="--", label="LDPC correction MATLAB")
plt.semilogy(df.EbN0dB, df.ber, marker="o", label="Cython Decoder (latest)")
plt.semilogy(dfpy.EbN0dB, dfpy.ber, marker="+", label="Python Decoder")
plt.semilogy(snr_range, p_acceptable, linestyle=":", label="Shannon limit")
plt.semilogy(snr_range, 0.5*(1-sp.special.erf(np.sqrt(10**(snr_range/10)/2))), label="No code")

plt.grid(True)
plt.legend()
plt.xlabel("$E_b/N_0$ [dB]")
plt.ylabel("Codeword BER $p_b$")
plt.title("BER vs SNR for a 1/2 rate code")

plt.show()
