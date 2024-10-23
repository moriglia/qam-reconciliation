# QAM Reconciliation

Reverse Reconciliation Softening experimental code.


## Library

The library contains mainly 3 parts:
1. a syndrome decoder
2. a noise mapper
3. some functions for the evaluation of mutual information

## Compilation

Either with `setuptools` by the command
```bash
python3 setup.py build_ext --inplace [--only SUBMODULE_NAME]
```
or by `make`


## Usage without installation

In order to use this code you have to add the project root to Python's search path, 
either by `export PYTHONPATH=$PYTHONPATH:/root/to/this/directory`
or by running every command with
```bash
PYTHONPATH=$PYTHONPATH:/path/to/this/directory python3 your_script.py [--your SCRIPT -s ARGUMENTS]
```

In your script you can `import` the extension classes for your simulations or experiments.

```python
from qamreconciliation import Matrix, Decoder, NoiseMapper, PAMAlphabet
import pandas as pd
import numpy as np

df = pd.read_csv("code.csv")
""" let's assume "code.csv" has 2 columns
"vid","cid"
0,0
0,2
2,9
...
where each line represents an edge between variable node "vid" and checknode "cid"
"""

# Create LDPC decoder
decoder = Decoder(df["vid"].to_numpy(), df["cid"].to_numpy())
matrix  = Matrix(df["vid"].to_numpy(), df["cid"].to_numpy())

# Create alphabet
bps     = 2 # bit per symbol
step    = 2 # constellation step
probabilities = np.array([0.15, 0.35, 0.35, 0.15], dtype=np.double)
pa      = PAMAlphabet(bps, step, probabilities)

# Create noise mapper
SNRdB   = 2
N0      = pa.variance * 10**(-SNRdB/10.0)
nm      = NoiseMapper(pa, N0/2)


# Frame parameters
N       = decoder.vnum       # number of variable nodes
K       = N - decoder.cnum   # number of information bits
N_symb  = N / bps

# Alice generates random symbols
x_i     = pa.random_symbols(N_symb) # index of constellation symbol
x       = np.array(pa.index_to_value(x_i)) # np.array is necessary because it outputs a TypedMemoryView

# Bob receives noisy symbols
y       = x + np.sqrt(N0/2)*np.random.randn(N_symb)
x_hat_i = nm.hard_decide_index(y) # decided symbols
word    = pa.demap_symbols_to_bits(x_hat_i)

# Bob generates the softening metric and the syndrome
n       = nm.map_noise(y, x_hat_i)   ## softening metric
synd    = mat.evaluate_syndrome(word)


# Alice uses her symbols to build the LLR for Bob's decisions
# exploiting the softening metric
llr     = nm.demap_lappr_array(n, x_i)

# Finally Alice decodes...
max_iterations = 50
(success, iterations, final_lappr) = decoder.decode(llr, synd, max_iterations)

print(success, iterations)

# count post-error-correction errors
error_count = 0
for i in range(K): # Only on the information bits
    if ((word[i]==0) ^ (final_lappr[i]>0)):
	    error_count += 1
print(error_count)

```


## Simulation script

The project is shipped with a simulation script `sims/sim_reconciliation.py` which has different parameters.
It can be invoked without adding the project root folder to Python's path just by calling it as a submodule
```bash
python3 -m sims.sim_reconciliation -h
```
The script accepts a few options to tweak the simulations:
```
$ python3 -m sims.sim_reconciliation -h
usage: decode [-h] [--out OUT] [--maxiter MAXITER] [--ferr-count-min FERR_COUNT_MIN] [--alpha ALPHA] [--simloops SIMLOOPS] [--snr SNR SNR] [--nsnr NSNR] [--bps BPS] [--hard] [--direct] [--configuration-base] edgefile

Evaluate BER for LDPC codes vs Raw BER

positional arguments:
  edgefile              CSV with a 'vid' and a 'cid' columns representing an edge per line

options:
  -h, --help            show this help message and exit
  --out OUT
  --maxiter MAXITER     Maximum number of iterations for the decoder
  --ferr-count-min FERR_COUNT_MIN
                        Minimum number of frame errors for early exit
  --alpha ALPHA         Extra multiplicative coefficient for the LLR
  --simloops SIMLOOPS   Number of frames per SNR point
  --snr SNR SNR         Initial and final SNR [dB] values of the range to evaluate the BER at
  --nsnr NSNR           Number of equally spaced SNR [dB] points to evaluate the BER at
  --bps BPS             Bit Per Symbol (=log_2(PAM Order))
  --hard                Simulate hard reverse reconciliation
  --direct              Simulate the soft direct reconciliation, overrides '--hard'
  --configuration-base  Instead of the Alternating configuration, use the Base configuration
```

This script outputs a CSV file with the following columns:
1. "EsN0dB" : SNR in dB
2. "ber" : Bit Error Rate
3. "fer" : Frame Error Rate
4. "iters" : average number of iterations to successfully decode a frame
