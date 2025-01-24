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
import fnmatch
import sys
from cleanup_cp import cleanup_cp

def get_full_list_of_files(root_dir, mask, base_dir=None):
    ans = list()
    if base_dir is None:
        base_dir = root_dir
    for r, _, f in os.walk(root_dir):
        for fn in f:
            if fnmatch.fnmatch(fn, mask):
                ans.append(os.path.relpath(os.path.join(r, fn), base_dir))
    return ans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default=None)
    parser.add_argument('--push', action='store_true', default=False)
    parser.add_argument('--pack', action='store_true', default=False)
    parser.add_argument('--ckeckpoints-mask', default='*.pth', help=r'Glob mask for checkpoints')
    parser.add_argument('-r', default=None, help='Name of remote of DVC for storing checkpints')
    parser.add_argument('--exclude-keys', default=["best_loss", "optimizer"], nargs="+", help='List of keys for exclusion from checkpoints before storing')
    args = parser.parse_args()

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    models_path = os.path.join(base_path, 'models', args.model_name)            
    
    cp_list = get_full_list_of_files(models_path, args.ckeckpoints_mask, base_dir=base_path)
    if len(args.exclude_keys) > 0 and len(args.exclude_keys[0]) > 0:
        for fn in cp_list:
            cleanup_cp(fn, fn, args.exclude_keys)
        
    dvc_list = [x + ".dvc" for x in cp_list]
       
    
    python_bin = os.path.dirname(sys.executable)
    dvc_path = os.path.join(python_bin, 'dvc')

    cmds = [
        f'{dvc_path} add ' + ' '.join(cp_list),
        f'{dvc_path} commit ' + ' '.join(dvc_list)
    ]
    if args.push:
        s = f'{dvc_path} push ' + ' '.join(dvc_list)
        if args.r is not None:
            s += f' -r {args.r}'
        cmds.append(s)

    os.system(' && '.join(cmds))
    os.system(f'git add ' + ' '.join(dvc_list))

    if args.pack:
        path = os.path.join(base_path, f'{args.model_name}.tgz')
        os.system(f'tar -cvzf {path} ' +  ' '.join(cp_list))


if __name__ == '__main__':
    main()
