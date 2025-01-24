# The copyright in this software is being made available under the BSD
# License, included below. This software may be subject to other third party
# and contributor rights, including patent rights, and no such rights are
# granted under this license.
#
# Copyright (c) 2010-2022, ITU/ISO/IEC
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
# be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import os
import numpy as np
from .reduce_z_distributions import ZReduction
from src.codec.entropy_coding.lib_wrappers.mans.ec_lib_mans import ECLibMans

def write_line2file(fn: str, f_t: str, data: np.ndarray):
    with open(fn, f_t) as f:
        f.write(",".join([str(x) for x in data.tolist()]) + "\n")

def main(cp_dir: str) -> None:
    zr = ZReduction()
    freqs_all = zr.collect_freq(cp_dir)
    
    unique_list, redidx_list = zr.convert_dist_to_indexies(freqs_all, freqs_all)
    zr.store_reduced_indexies(cp_dir, unique_list, redidx_list)

    cdfs = np.array(unique_list)
    l_cou = cdfs.shape[0]
    mem = np.ndarray((10000))
    for ln in range(l_cou):
        enc = ECLibMans()
        enc.encode_init(mem)
        transactions = enc.get_z_transactions(cdfs[ln])
        stateNext = (transactions >> 16) & 255
        symbols = transactions & 255
        nbits = (transactions >> 24) & 255
        enc.encode_term()
        
        file_t = 'w' if ln == 0 else 'a'
        write_line2file(os.path.join(cp_dir, "transition_table_z_symbol.csv"), file_t, symbols)
        write_line2file(os.path.join(cp_dir, "transition_table_z_nBits.csv"), file_t, nbits)
        write_line2file(os.path.join(cp_dir, "transition_table_z_stateNext.csv"), file_t, stateNext)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp_dir", type=str, default="models/VM_common_int",
                        help="path to a directory with checkpoints")
    args = parser.parse_args()
    main(args.cp_dir)

