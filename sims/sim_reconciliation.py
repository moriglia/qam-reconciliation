# -*- mode: python ; gud -*-
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
    import pandas as pd
    import numpy as np
    from qamreconciliation import Matrix, Decoder, PAMAlphabet
    from sims.reconciliation import simulate_direct_snr_dB, simulate_softening_snr_dB, simulate_hard_reverse_snr_dB
    from parfor import parfor
    
    parser = argparse.ArgumentParser(
        prog="decode",
        description="Evaluate BER for LDPC codes vs Raw BER"
    )
    
    parser.add_argument("edgefile", help="CSV with a 'vid' and a 'cid' columns representing an edge per line")
    parser.add_argument("--out", default="out.csv")

    parser.add_argument("--maxiter",      default = 50, type=int, help="Maximum number of iterations for the decoder")
    parser.add_argument("--ferr-count-min", default = 100, type=int, help="Minimum number of frame errors for early exit")
    parser.add_argument("--alpha", type=float, default=1.0, help="Extra multiplicative coefficient for the LLR")
    parser.add_argument("--simloops",     default = 5000, type=int, help="Number of frames per SNR point")
    parser.add_argument("--snr", type=float, nargs=2, default=[0,5], help="Initial and final SNR [dB] values of the range to evaluate the BER at")
    parser.add_argument("--nsnr", type=int, default=11, help="Number of equally spaced SNR [dB] points to evaluate the BER at")
    parser.add_argument("--bps", type=int, default=2, help="Bit Per Symbol (=log_2(PAM Order))")

    parser.add_argument("--hard", action="store_true", help="Simulate hard reverse reconciliation")
    parser.add_argument("--direct", action="store_true", help="Simulate the soft direct reconciliation, overrides '--hard'")
    parser.add_argument("--configuration-base", action="store_true", help="Instead of the Alternating configuration, use the Base configuration")
    
    args = parser.parse_args()

    
    edge_df = pd.read_csv(args.edgefile)
    N_symb = edge_df.vid[0] // args.bps

    EsN0dB = np.linspace(args.snr[0], args.snr[1], args.nsnr)
    # final_ber = []


    if (args.direct):
        @parfor(EsN0dB)
        def final_ber(snr_dB):
            dec = Decoder(edge_df.vid[1:].to_numpy(), edge_df.cid[1:].to_numpy())
            mat = Matrix( edge_df.vid[1:].to_numpy(), edge_df.cid[1:].to_numpy())
            pa  = PAMAlphabet(args.bps, 2)
            return simulate_direct_snr_dB(snr_dB, dec, mat, pa,
                                          args.maxiter,
                                          args.simloops,
                                          args.ferr_count_min)
    elif (args.hard):
        @parfor(EsN0dB)
        def final_ber(snr_dB):
            dec = Decoder(edge_df.vid[1:].to_numpy(), edge_df.cid[1:].to_numpy())
            mat = Matrix( edge_df.vid[1:].to_numpy(), edge_df.cid[1:].to_numpy())
            pa  = PAMAlphabet(args.bps, 2)
            return simulate_hard_reverse_snr_dB(snr_dB, dec, mat, pa,
                                          args.maxiter,
                                          args.simloops,
                                          args.ferr_count_min)
        
    else:
        @parfor(EsN0dB)
        def final_ber(snr_dB):
            dec = Decoder(edge_df.vid[1:].to_numpy(), edge_df.cid[1:].to_numpy())
            mat = Matrix( edge_df.vid[1:].to_numpy(), edge_df.cid[1:].to_numpy())
            pa  = PAMAlphabet(args.bps, 2)
            nmconfig = np.zeros(pa.order, dtype=np.uint8)
            if (not args.configuration_base):
                nmconfig[1::2] = 1
            
            return simulate_softening_snr_dB(snr_dB, dec, mat, pa,
                                             nmconfig,
                                             args.maxiter,
                                             args.simloops,
                                             args.ferr_count_min,
                                             args.alpha)

    
    df = pd.DataFrame(final_ber,
                      columns=['EsN0dB',
                               'ber',
                               'fer',
                               'iters'])
    
    df.to_csv(args.out)
