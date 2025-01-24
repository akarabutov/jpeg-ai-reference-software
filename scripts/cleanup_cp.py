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

import torch


def update_cp(cp, str_exclude):
    ans = dict()
    for k, v in cp.items():
        for exc_v in str_exclude:
            if k == exc_v:
                v = None
                k = None
        if k is not None:
            if isinstance(v, dict):
                v = update_cp(v, str_exclude)
            ans[k] = v
    return ans

def cleanup_cp(input_fn, output_fn, exclude):
    inp = torch.load(input_fn)

    ans = update_cp(inp, exclude)

    torch.save(ans, output_fn)
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_fn', type=str, default=None)
    parser.add_argument('output_fn', type=str, default=None)
    parser.add_argument('--exclude',
                        type=str,
                        nargs='+',
                        default=[],
                        help='List of keys for exclusion')
    args = parser.parse_args()
    cleanup_cp(args.input_fn, args.output_fn, args.exclude)
    print(f'Store data to file {args.output_fn}')


if __name__ == '__main__':
    main()
