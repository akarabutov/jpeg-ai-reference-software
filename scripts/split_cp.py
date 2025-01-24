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
import torch
import argparse
import re
from typing import Dict, List


def split_cp(state_data: Dict[str, torch.Tensor], output_fn: str, op_list: List[str], output_base_dir: str, output_op: List[str] = None, verbose: bool = False, fields_mask:str=".*") -> None:
    """Split checkpoints state dict and store its components to different directories

    Args:
        state_data (Dict[str, torch.Tensor]): input state dict
        output_fn (str): a name of the output file
        op_list (List[str]): list with operation points
        output_base_dir (str): output to the base directory
        output_op (List[str], optional): list with name of directories for operation points in the same order as in op_list. Additionally you can set a name of a directory with the common parts of the codec, by default it is "VM_common". Defaults to None and all directories have name "VM_<op>".
        verbose (bool): vervose mode. Default: False
    """
    net_parts = ["encoder", "decoder"]
    f_re = re.compile(fields_mask)
    # Store encoder/decoder parts
    for i, op in enumerate(op_list):
        op_dir_name = f"VM_{op}" if output_op is None else output_op[i]
        output_state = dict()
        op_substr = f".coders.{op}_"
        for sn, sv in state_data.items():
            enc_dec = [sn.startswith(n) for n in net_parts]
            enc_dec = any(enc_dec)
            if op_substr in sn and enc_dec and len(f_re.findall(sn))>0:
                sn_list = sn.split('.')
                sn_new = ".".join(sn_list[3:]) 
                sn_base = sn_list[0]

                if sn_base not in output_state:
                    output_state[sn_base] = dict()
                output_state[sn_base][sn_new] = sv
            
        # Add epoch to all sub-checkpoints
        if "epoch" in state_data:
            for cp_n in output_state.keys():
                output_state[cp_n]['epoch'] = state_data['epoch']
                
        # Store data to output files
        output_dir = os.path.join(output_base_dir, op_dir_name)
        os.makedirs(output_dir, exist_ok=True)
        for net_part_name, op_v in output_state.items():
            if len(op_v) > 0:
                full_output_fn = os.path.join(output_dir, f"{net_part_name}_{output_fn}")
                torch.save(op_v, full_output_fn)
                if verbose:
                    print(f"Stored {net_part_name} data to a file {full_output_fn}")
            
    # Store common parts
    op_dir_name = f"VM_common" if output_op is None else output_op[-1]
    output_dir = os.path.join(output_base_dir, op_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    for comp in ['Y', 'UV']:
        output_state = dict()
        cur_ending = f"_{comp}"
        for sn, sv in state_data.items():
            enc_dec = [sn.startswith(n) for n in net_parts]
            enc_dec = any(enc_dec)
            sn_list = sn.split('.')
            sn_base = sn_list[0]
            if not enc_dec and (sn_base.endswith(cur_ending) or "_" not in sn_base) and len(f_re.findall(sn))>0:
                sn_list[0]=sn_list[0].replace(cur_ending, "")
                sn_new = ".".join(sn_list) 
                output_state[sn_new] = sv
        if len(output_state) > 0:
            full_output_fn = os.path.join(output_dir, f"{comp}_{output_fn}")
            torch.save(output_state, full_output_fn)
            if verbose:
                print(f"Stored common part to a file {full_output_fn}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_fn', type=str, default=None, help=r"Path to the input checkpoint")
    parser.add_argument('output_base_dir', type=str, default=None, help=r"Path to output base directory for models")
    parser.add_argument(
        '--op_list',
        type=str,
        nargs='+',
        default=["hop", "bop", "sop"],
        help=r'List of operation points to be extracted')
    parser.add_argument(
        '--fields_mask',
        type=str,
        default=".*",
        help=r'Mask for fields to be extracted')    
    parser.add_argument('--verbose', default=False, action='store_true', help=r"Verbose mode")
    parser.add_argument('--prefix', default="VM_", type=str, help=r"prefix of output model's directories")
    args = parser.parse_args()
    cp_arr = torch.load(args.input_fn)
    
    prefix = args.prefix
    output_op = [f"{prefix}{x}" for x in args.op_list]
    output_op.append(f"{prefix}common")
    
    split_cp(cp_arr, os.path.basename(args.input_fn), args.op_list, args.output_base_dir, output_op, args.verbose, args.fields_mask)


if __name__ == "__main__":
    main()