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
from typing import List

def update_cp_elem(cp, old_name: List, new_name: List):
    if len(old_name) == 0 or len(new_name) == 0:
        return
    old_n = old_name[0]
    new_n = new_name[0]
    if old_n != new_n:
        cp[new_n] = cp[old_n]
        del cp[old_n]
    concat_old_name = ".".join(old_name)
    if concat_old_name in cp:
        concat_new_name = ".".join(new_name)
        cp[concat_new_name] = cp[concat_old_name]
        del cp[concat_old_name]        
    else:
        update_cp_elem(cp[new_n], old_name[1:], new_name[1:])
    

def update_cp(cp, str_replaces):
    for os, ns in str_replaces.items():
        update_cp_elem(cp, os.split("."), ns.split("."))
    return cp


def list2dict(lst):
    assert len(lst) % 2 == 0
    ans = dict()
    for i in range(0, len(lst), 2):
        ans[lst[i]] = lst[i + 1]
    return ans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_fn', type=str, default=None)
    parser.add_argument('output_fn', type=str, default=None)
    parser.add_argument(
        '--rename',
        type=str,
        nargs='+',
        default=[],
        help='Patterns of names for renameing and the new values in form: <pattern> <new_name>')
    args = parser.parse_args()

    inp = torch.load(args.input_fn)

    a = list2dict(args.rename)
    print(a)
    ans = update_cp(inp, a)

    torch.save(ans, args.output_fn)
    print(f'Store data to file {args.output_fn}')


if __name__ == '__main__':
    main()
