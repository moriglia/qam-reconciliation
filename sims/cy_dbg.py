from qamreconciliation.decoder_py import Decoder as PyDecoder
from qamreconciliation.decoder_cy import Decoder as CyDecoder
from qamreconciliation import Matrix
from galois import GF2
import numpy as np
import pandas as pd
import time
import math


tannerGraph = pd.read_csv("sims/dvbs2ldpc0.500.csv")

mat = Matrix(tannerGraph.vid[1:].to_numpy(),
             tannerGraph.cid[1:].to_numpy())
N = mat.vnum

pydec = PyDecoder(tannerGraph, True)
cydec = CyDecoder(tannerGraph.vid[1:].to_numpy(),
                  tannerGraph.cid[1:].to_numpy())

word = GF2.Random(N)
synd = mat.eval_syndrome(word)

snrdB=0
snr=10**(snrdB/10)
N0 = 1/snr
sigma = np.sqrt(N0)

rcvd = (1.0-2.0*np.array(word, dtype=np.ubyte)) + sigma*np.random.randn(N)
llr = np.array(2*np.log2(np.e)/N0*rcvd, dtype=np.double)

t0 = time.time()
(_, cyit, cylappr) = cydec.decode(llr, synd, 20)
t1 = time.time()
(_, pyit, pylappr) = pydec.decode(llr, synd, 20)
t2 = time.time()

cylappr = np.array(cylappr)
pylappr = np.array(pylappr)

print((np.sign(pylappr) != np.sign(cylappr)).sum())
delta = np.abs(pylappr - cylappr)
minval = np.array([min(np.abs(pylappr[i]), np.abs(cylappr[i])) for i in range(N)])
reldelta = delta[delta != math.inf]/minval[delta != math.inf]

print(reldelta.max(), reldelta.mean())
print(np.array(pylappr==math.inf).sum())
print(t1-t0, t2-t1)
print(cyit, pyit)
