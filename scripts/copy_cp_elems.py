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

import torch
from typing import List, Dict, Any


def update_dict_recursively(orig_dict: Dict[str, Any], new_dict: Dict[str, Any]) -> Dict[str, Any]:
    for key, val in new_dict.items():
        if isinstance(val, dict):
            tmp = update_dict_recursively(orig_dict.get(key, {}), val)
            orig_dict[key] = tmp
        else:
            orig_dict[key] = new_dict[key]
    return orig_dict


def update_element(src: Dict, dst: Dict, elem_url: List):
    elem_key = elem_url[0]
    concat_el = ".".join(elem_url)
    if len(elem_url) == 1 or (concat_el in src):
        if elem_key in src:
            dst[elem_key] = update_dict_recursively(src[elem_key], dst[elem_key]) 
        else:
            for k in src.keys():
                if k.startswith(elem_key):
                    if isinstance(src[k], dict):
                        dst[k] = update_dict_recursively(src[k], dst[k]) 
                    else:
                        dst[k] = src[k]
    else:
        update_element(src[elem_key], dst[elem_key], elem_url[1:])

def copy_checkpoints(input1_fn: str, input2_fn: str, output_fn: str, copy_list: List) -> None:
    input1_cp = torch.load(input1_fn)
    ans = torch.load(input2_fn)
    
    for el in copy_list:
        print(f"Copy {el} element(s).")
        update_element(input1_cp, ans, el.split("."))
    
    torch.save(ans, output_fn)


def main():
    parser = argparse.ArgumentParser(description=r'Copy elements from one checkpoints to another one')
    parser.add_argument('input1_fn', type=str, default=None)
    parser.add_argument('input2_fn', type=str, default=None)
    parser.add_argument('output_fn', type=str, default=None)
    parser.add_argument('--copy',
                        type=str,
                        nargs='+',
                        default=[],
                        help=r'List of elements to be copied from a file "input1_fn" to structure of "input2_fn" and all of them will be stored to "output_fn". If the "input2_fn" have elements with the same names, they will be overwritten.')
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_fn)
    os.makedirs(output_dir, exist_ok=True)

    copy_checkpoints(args.input1_fn, args.input2_fn, args.output_fn, args.copy)

    print(f'Store data to file {args.output_fn}')


if __name__ == '__main__':
    main()
