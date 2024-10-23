from matplotlib import pyplot as plt
import pandas as pd
import scipy as sp
import numpy as np
from qamreconciliation import bicm, alphabet, NoiseMapper
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--bps", type=int, default=1)
parser.add_argument("--file", nargs=2, action='append')
parser.add_argument("--title", default="Use '--title \"My title\"' to set the title")
parser.add_argument("--rate", type=float, default=1)
parser.add_argument("--xlabel",
                    type=str,
                    default="Use '--xlabel \"\\\$E\_b/N\_0\\\$\"' option to set the label to \"$E_b/N_0$ [dB]\"")
parser.add_argument("--ylabel",
                    type=str,
                    default="Use '--ylabel \"\\\$p\_b\\\$\"' option to set the label to \"$p_b$\"")

args = parser.parse_args()

print(args.file)


df = []
legend = []
for file_legend in args.file:
    df.append(pd.read_csv(file_legend[0]))
    legend.append(file_legend[1])


bit_per_symbol = args.bps
pamorder = 1<<bit_per_symbol

al = alphabet.PAMAlphabet(bit_per_symbol, 2)
s_to_b = bicm.generate_table_s_to_b(bit_per_symbol)
n_err = bicm.generate_error_number_table(s_to_b)


snrdb_range = np.linspace(-5, 15, 41)
N0 = 10**(-snrdb_range/10) * al.variance

p_b = np.empty_like(N0)
for i in range(len(p_b)):
    noiseMapper = NoiseMapper(al, N0[i])
    p_b[i] = 0
    for tx in range(pamorder):
        for rx in range(pamorder):
            p_b[i] += noiseMapper.fwrd_transition_probability[rx, tx] \
                * al.probabilities[tx] * n_err[rx, tx]

p_b /= bit_per_symbol


rate_bit_shift = -10*np.log10(args.rate*bit_per_symbol)

for i in range(len(df)):
    plt.semilogy(df[i].EsN0dB + rate_bit_shift, df[i].ber, label=legend[i])

plt.semilogy(snrdb_range, p_b, linestyle=":", label="Uncoded error rate")
plt.grid(True, which="both")
plt.legend(fontsize=18)
# plt.xlabel("$E_s/N_0$ [dB]")
# plt.ylabel("Codeword BER $p_b$")
plt.xlabel(args.xlabel, fontsize=20)
plt.ylabel(args.ylabel, fontsize=20)
plt.title(args.title, fontsize=22)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
