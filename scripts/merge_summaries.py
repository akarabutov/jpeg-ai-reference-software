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

import os
from pathlib import Path
from argparse import ArgumentParser
from append_summ2xlsm import append_summary

def get_args():
    parser = ArgumentParser()
    parser.add_argument('BASE_DIR', type=str, help="Path to base_directory")
    parser.add_argument('--prefix', type=str, default=None, help="Prefix for sim names")
    parser.add_argument('--fn-prefix', type=str, default="merge_summary", help="Prefix for output files name")
    parser.add_argument('--columns-with-data', type=int, default=19, help="Number of columns with data")
    parser.add_argument('--post-columns', type=int, default=13, help="Number of tabs after each columns")
    parser.add_argument('--max-results-number', type=int, default=13, help="Number of results in each output text file")
    return parser.parse_args()

def get_summary_data(path, column_number):
    ans = list()
    all_match = True
    with open(path, "r") as f:
        for l in f:
            d = [x.strip() for x in l.split('\t')]
            all_match = all_match and (d[-1].lower() == "match")
            if len(d) < column_number:
                d += ["None"] * (column_number - len(d))
            ans.append("\t".join(d))
    return ans, all_match

def get_max_lines_count(summ_in):
    ans = 0
    for s in summ_in:
        ans = max(ans, len(s))
    return ans

def merge_summaries(summ_in, delim_count, data_column_num):
    ans = list()
    delim = '\t' * delim_count
    mlc = get_max_lines_count(summ_in)
    fill_str = "\t".join(["None"] * data_column_num)
    for i in range(mlc):
        l_cur = list()
        for j in range(len(summ_in)):
            s = summ_in[j]
            cur_l = s[i] if i < len(s) else fill_str
            l_cur.append(cur_l)
        l_str = delim.join(l_cur)
        ans.append(l_str)
    return ans

    
def process_summaries(BASE_DIR: str, 
                      prefix: str = None, 
                      fn_prefix: str = "merge_summary", 
                      columns_with_data: int = 19,
                      post_columns: int = 13,
                      max_results_number: int = 13,
                      anchor: str = None) -> None:
    output_summs = list()
    paths = sorted(Path(BASE_DIR).iterdir(), key=os.path.getmtime)
    pref_init = prefix.replace("_", "-") if prefix is not None else None
    prefix = "" if pref_init is None else (pref_init + "-")
    for full_path_d in paths:
        d = os.path.basename(full_path_d)
        if os.path.isdir(full_path_d):
            enc_sum_path = os.path.join(full_path_d, "summary.txt")
            dec_sum_path = os.path.join(full_path_d, "summary_dec.txt")
            has_enc_summary = os.path.exists(enc_sum_path)
            has_dec_summary = os.path.exists(dec_sum_path)
            if has_enc_summary:
                enc_data, enc_match = get_summary_data(enc_sum_path, columns_with_data)
                dec_data = None
                test_name = d.replace("_", "-")
                if not enc_match and has_dec_summary:
                    dec_data, _ = get_summary_data(dec_sum_path, columns_with_data)
                    enc_data[0] = prefix + test_name + "-ENC_" + enc_data[0]
                    dec_data[0] = prefix + test_name + "-DEC_" + dec_data[0]
                else:
                    enc_data[0] = prefix + test_name + "_" + enc_data[0]
                    
                output_summs.append(enc_data)
                if dec_data is not None:
                    output_summs.append(dec_data)                    
    for i, s_start in enumerate(range(0, len(output_summs), max_results_number)):
        output_fn = f"{fn_prefix}_{i}"
        output_sum_path = os.path.join(BASE_DIR, f"{output_fn}.txt")
        with open(output_sum_path, 'w') as f:
            m_sum = merge_summaries(output_summs[s_start:min(s_start+max_results_number, len(output_summs))],
                                    delim_count=post_columns, data_column_num=columns_with_data)
            for l in m_sum:
                f.write(l + "\n")
        append_summary(output_sum_path, os.path.join(BASE_DIR, f"{output_fn}.xlsm"), anchor=anchor)
        
def main():
    args = get_args()
    process_summaries(**vars(args))    


if __name__ == '__main__':
    main()