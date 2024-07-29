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
import logging
import os
import re


def get_ctrl_points_md5(s):
    r_ctrl = re.compile('Logger\[(?P<module_name>.*)\] .*: The (?P<cp_id>.*) control point is (?P<cp_md5>.*)')
    ans = dict()
    for a in r_ctrl.finditer(s):
        module_name = a.group('module_name')
        cp_id = a.group('cp_id')
        cp_md5 = a.group('cp_md5')
        mm = ans.get(module_name, dict())
        mi = mm.get(cp_id, list())
        mi.append(cp_md5)
        mm[cp_id] = mi
        ans[module_name] = mm
    return ans        

def compare_cp_md5(cp1, cp2):
    ans = True
    if len(cp1) > len(cp2):
        cp1, cp2 = cp2, cp1
        
    for k in cp1.keys():
        if k not in cp2:
            print(f"Cannot find module {k}")
            ans = False
        else:
            ccp1 = cp1[k]
            ccp2 = cp2[k]
            for kk in ccp1.keys():
                if kk not in ccp2:
                    print(f"Cannot find control point {kk} for module {k}")
                    ans = False
                else:
                    tmp = False
                    for cccp1 in ccp1[kk]:
                        if cccp1 in ccp2[kk]:
                            tmp = True
                    ans = ans and tmp
                    if not tmp:
                        print(f"Mismatch for the {kk} control point in a module {k}: {ccp1[kk]}, {ccp2[kk]}")
    return ans

def compare_files(file1, file2):
    print(f'Start comparing {file1} and {file2}')
    r = re.compile('MD5: \((?P<md5_a>[0-9a-fA-F]+),\s+(?P<md5_b>[0-9a-fA-F]+),\s+(?P<md5_c>[0-9a-fA-F]+)\)')

    with open(file1, 'r') as f:
        txt = f.read()
        v1 = r.findall(txt)
        cp1 = get_ctrl_points_md5(txt)
    with open(file2, 'r') as f:
        txt = f.read()
        v2 = r.findall(txt)
        cp2 = get_ctrl_points_md5(txt)

    for fr1, fr2 in zip(v1, v2):
        img_match = fr1 == fr2
        
    cp_match = compare_cp_md5(cp1, cp2)
    img_match = img_match and cp_match

    if img_match:
        print(f'Files are matched')
    else:
        print(f'MISMATCH between files {file1} and {file2}')


def get_files_list(f):
    if os.path.isfile(os.path.abspath(f)):
        return [os.path.basename(f)], [f]
    elif os.path.isdir(f):
        A = [x for x in os.listdir(f) if x.endswith('.yuv')]
        return A, [os.path.join(f, x) for x in A]
    else:
        raise NotImplementedError


def compare(args=None):
    # Import default config file name
    ap = argparse.ArgumentParser('Tool for comparing YUV files')
    logger_opts = {
        'error': logging.ERROR,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'critical': logging.CRITICAL,
        'debug': logging.DEBUG
    }
    ap.add_argument('--log-level',
                    required=False,
                    default=[*logger_opts][1],
                    choices=[*logger_opts])
    ap.add_argument('--i1', type=str, help='Path to file/directory of encoder')
    ap.add_argument('--i2', type=str, help='Path to file/directory of decoder')
    ap.add_argument('--regex', type=str, default='.*', help='Regex filter')

    args, _ = ap.parse_known_args(args)

    print(vars(args))

    fl1, ffl1 = get_files_list(args.i1)
    fl2, ffl2 = get_files_list(args.i2)

    flt_names = re.compile(args.regex)

    if len(fl1) < len(fl2):
        fl1, ffl1, fl2, ffl2 = fl2, ffl2, fl1, ffl1

    #for (f1, f2) in zip(fl1, fl2):
    for f1, ff1 in zip(fl1, ffl1):
        if flt_names.match(f1):
            if f1 in fl2:
                ff2 = ffl2[fl2.index(f1)]
                compare_files(ff1, ff2)
            else:
                print(f'Did file corresponding file for {f1}')

if __name__ == "__main__":
    compare()