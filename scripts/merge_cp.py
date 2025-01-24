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

from src.codec.common.utils import update_dict_recursively


def merge_checkpoints(remove_list, exclude_list, *args):
    ans = dict()
    for cp_fn in args:
        print(f'Process file {cp_fn}...')
        cp = torch.load(cp_fn)
        print(f'{cp_fn} {cp.keys()}')
        # Remove keys
        for rk in remove_list:
            if rk in cp.keys():
                print(f'Remove key {rk} from cp in file {cp_fn}')
                del cp[rk]

        # Exclude blocks
        if len(ans) != 0:
            for e in exclude_list:
                if e in cp.keys():
                    print(f'Remove key {e} from cp in file {cp_fn}')
                    del cp[e]
        ans = update_dict_recursively(ans, cp)
    return ans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_fn', type=str, default=None)
    parser.add_argument('checkpoints', type=str, nargs='+', default=[])
    parser.add_argument('--exclude',
                        type=str,
                        nargs='+',
                        default=[],
                        help='List of modeles on first level, which will be excluded from merging')
    parser.add_argument('--remove',
                        type=str,
                        nargs='+',
                        default=[],
                        help='List of keys for removing')
    args = parser.parse_args()

    merged_cp = merge_checkpoints(args.remove, args.exclude, *args.checkpoints)

    output_dir = os.path.dirname(args.output_fn)
    os.makedirs(output_dir, exist_ok=True)

    torch.save(merged_cp, args.output_fn)
    print(f'Store data to file {args.output_fn}')


if __name__ == '__main__':
    main()
