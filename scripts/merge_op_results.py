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
from argparse import ArgumentParser
from append_summ2xlsm import append_summary
from merge_summaries import merge_summaries, get_summary_data


def get_args(args = None):
    cur_file_path=os.path.dirname(os.path.abspath(__file__))    
    parser = ArgumentParser()
    parser.add_argument('BASE_DIR', type=str, help="Path to base_directory of results of oper.points")
    parser.add_argument('--prefix', type=str, default=None, help="Prefix for sim names")
    parser.add_argument('--fn-prefix', type=str, default="merge_summary", help="Prefix for output files name")
    parser.add_argument('--columns-with-data', type=int, default=19, help="Number of columns with data")
    parser.add_argument('--post-columns', type=int, default=13, help="Number of tabs after each columns")
    parser.add_argument('--max-results-number', type=int, default=13, help="Number of results in each output text file")
    parser.add_argument('--template', type=str, default=os.path.join(cur_file_path, os.pardir, "docs", "template.xlsm"), help="Path to template Excel file")
    parser.add_argument('--start-row', type=int, default=3, help="Row of starting cell")
    parser.add_argument('--start-col', type=int, default=32, help="Column of starting cell")
    parser.add_argument('--flt', type=str, default=None, help="Filter for directories")
    parser.add_argument('--anchor', type=str, default=None, help="Set anchor on summary page")
    return parser.parse_known_args(args)

def process_summaries(BASE_DIR: str, 
                      prefix: str = None, 
                      fn_prefix: str = "merge_summary", 
                      columns_with_data: int = 19,
                      post_columns: int = 13,
                      max_results_number: int = 13,
                      anchor: str = None,
                      flt: str = None,
                      template: str = None,
                      start_row: int = 3,
                      start_col: int = 32) -> None:
    
    output_summs = list()
    output_sums_dict = dict()
    sum_fns = ['summary.txt', 'summary_dec.txt']   
    sum_suffix = ['-ENC', '-DEC']
    pref_init = prefix.replace("_", "-") if prefix is not None else None
    prefix = "" if pref_init is None else (pref_init + "-")
    
    for root, _, files in os.walk(BASE_DIR):
        rel_path = os.path.relpath(root, BASE_DIR)
        if flt is not None and flt not in rel_path:
            continue
        #test_name = os.path.basename(root).replace("_", "-")
        base_dir = os.path.dirname(root)
        #op_name = os.path.basename(base_dir).replace("_", "-")
        task_name = rel_path.replace("_", "-").replace("/", ":")
        tmp_data = dict()
        for sum_fn in sum_fns:
            if sum_fn in files:
                res = dict()
                path = os.path.join(root, sum_fn)
                res["data"], res["match"] = get_summary_data(path, columns_with_data)
                tmp_data[sum_fn] = res
        if tmp_data.get(sum_fns[0], dict()).get('match', False) and sum_fns[1] in tmp_data:
            tmp_data.pop(sum_fns[1])
        
        if len(tmp_data) > 0:
            use_suffix = len(tmp_data) > 1
            for sum_fn, sum_suf in zip(sum_fns, sum_suffix):
                if sum_fn in tmp_data:
                    name = prefix + task_name + (sum_suf if use_suffix else "") 
                    tmp_data[sum_fn]["data"][0] = name + "_" + tmp_data[sum_fn]["data"][0]
                    #output_summs.append(tmp_data[sum_fn]["data"])
                    output_sums_dict[task_name] = tmp_data[sum_fn]["data"]
                    
    for k in sorted(output_sums_dict.keys()):
        output_summs.append(output_sums_dict[k])

    for i, s_start in enumerate(range(0, len(output_summs), max_results_number)):
        output_fn = f"{fn_prefix}_{i}"
        output_sum_path = os.path.join(BASE_DIR, f"{output_fn}.txt")
        with open(output_sum_path, 'w') as f:
            m_sum = merge_summaries(output_summs[s_start:min(s_start+max_results_number, len(output_summs))],
                                    delim_count=post_columns, data_column_num=columns_with_data)
            for l in m_sum:
                f.write(l + "\n")
        append_summary(output_sum_path, 
                       os.path.join(BASE_DIR, f"{output_fn}.xlsm"), 
                       anchor=anchor, 
                       template=template,
                       start_row=start_row,
                       start_col=start_col)
            

def main():
    args, _ = get_args()
    process_summaries(**vars(args))    


if __name__ == '__main__':
    main()
                
            
            