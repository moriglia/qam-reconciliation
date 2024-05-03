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


def phi_root_locus(p_b, epsilon, R):
    return h2(p_b) - 1 + (1 - h2(epsilon))/R


df = pd.read_csv("res_dvbs2ldpc0.750_cython.csv")
dfp = pd.read_csv("res_dvbs2ldpc0.750_python.csv")
dfm = pd.read_csv("matlab/res_dvbs2ldpc0.750_matlab.csv", header=None)
# dfmat = pd.read_csv("matlab/res_dvbs2_matlab.csv")


ber_range = np.linspace(0.01, 0.1, 91)
p_acceptable = np.empty(91, dtype=np.float64)

for i in range(len(ber_range)):
    try:
        p_acceptable[i] = sp.optimize.brentq(
            phi_root_locus,
            a=0, b=0.5,
            args=(ber_range[i], 0.75)
        )
    except ValueError as ve:
        print(ve)
        p_acceptable[i] = 0
        
# matlab_ber = pd.read_csv("matlab/res_dvbs2ldpc_0.500_matlab.csv", header=None)

# print(matlab_ber)

plt.semilogy(df.f, df.ber, marker="x", label="Cython Decoder")
plt.semilogy(ber_range, p_acceptable, linestyle="-.", label="Shannon limit")
plt.semilogy(dfp.epsilon, dfp.ber, marker="o", label="Python Decoder")
plt.semilogy(dfp.epsilon, dfm[:], linestyle="--", label="Matlab Decoder")
# plt.semilogy(epsilon_range, p_acceptable, linestyle="-.", label="Min achievable $p_b$")
# plt.semilogy(matlab_ber[0], matlab_ber[1], linestyle="--", label="LDPC correction MATLAB")
# plt.semilogy(df.EbN0dB, df.ber, marker="o", label="Cython Decoder (restricted domain)")

identity = 10**np.linspace(-5, -1, 41)
plt.semilogy(identity, identity, label="No code")

plt.grid(True)
plt.legend()
plt.xlabel("$f$")
plt.ylabel("Codeword BER $p_b$")
plt.title("BER vs flipping probability for a 3/4 rate code")

plt.show()
