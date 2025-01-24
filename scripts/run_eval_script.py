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
import sys
import argparse
from multiprocessing import current_process
from typing import List
from concurrent.futures import ProcessPoolExecutor as Pool
import subprocess
import copy
from typing import Dict, Any

def update_dict_recursively(orig_dict: Dict[str, Any], new_dict: Dict[str, Any]) -> Dict[str, Any]:
    for key, val in new_dict.items():
        if isinstance(val, dict):
            tmp = update_dict_recursively(orig_dict.get(key, {}), val)
            orig_dict[key] = tmp
        else:
            orig_dict[key] = new_dict[key]
    return orig_dict


def get_args() -> dict:
    cur_file_path=os.path.dirname(os.path.abspath(__file__))
    cfg_default=os.path.join(cur_file_path, os.pardir, "cfg", "eval", "tools_onoff.json")
    parser = argparse.ArgumentParser()
    parser.add_argument('output_base_dir', type=str, help="Output base directory")
    parser.add_argument('--cfg', type=str, default=cfg_default, help="Path to configuration file")
    parser.add_argument('--module', type=str, default="src.reco.scripts.eval", help="Module for simulation")
    parser.add_argument('--silent', action='store_true', default=False, help="Output base directory")
    return parser.parse_known_args()


def load_cfg(path) -> Dict:
    import json
    ans = dict()
    cfg_dir_path = os.path.dirname(os.path.abspath(path))
    
    with open(path, 'r') as f:
        cfg = json.load(f)
        sub_cfg_paths = cfg.pop("!include", list())
        for sub_cfg_path in sub_cfg_paths:
            sub_cfg_path = os.path.abspath(os.path.join(cfg_dir_path, sub_cfg_path)) if not os.path.isabs(sub_cfg_path) else sub_cfg_path
            sub_cfg = load_cfg(sub_cfg_path)
            ans = update_dict_recursively(ans, sub_cfg)
        ans = update_dict_recursively(ans, cfg)
    return ans


def config_gpu_list() -> List:
    gpu_list = list()

    # Get list with available GPUs
    try:
        import GPUtil
        gpu_ids = GPUtil.getAvailable(limit=float('inf'))
    except:
        gpu_ids = [-1]

    # List of enabled GPU
    gpu_strs = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if len(gpu_strs) == 0:
        gpu_list = gpu_ids
    else:
        for gpu_str in gpu_strs.split(','):
            gpu_str = gpu_str.strip()
            if len(gpu_str) > 0 and ((int(gpu_str) in gpu_ids) or (int(gpu_str) < 0)):
                gpu_list.append(int(gpu_str))

    return gpu_list


def set_gpu_id(gpu_ids: List) -> int:
    gpu_id_idx = 0
    if 'MainProcess' not in current_process().name:
        gpu_id_idx = int(current_process().name.split('-')[1]) -1
    gpu_id = gpu_ids[gpu_id_idx]
    env=None
    if gpu_id >= 0:
        env = dict()
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    return gpu_id
        
def compute(cfg):
    import sys
    python_path = sys.executable
    gpu_list = cfg.get('gpu_list', [])
    base_path = cfg.get('base_path', '')
    ouput_base_dir = cfg.get('ouput_base_dir', '')
    cfg_name = cfg.get('cfg_name', '')
    append_profile_config = cfg.get('append_profile_config', True)
    pre_args = cfg.get('pre_args', [])
    post_args = cfg.get('post_args', [])
    verbose = cfg.get('verbose', False)
    profile = cfg.get('profile', 'base')
    gpu_id = set_gpu_id(gpu_list)
    cmd = [
        python_path,
        "-m", cfg.get('module_name', '')
    ]
    cmd += pre_args
    output_dir = None
    for k,v in cfg.get('cfg', dict()).items():
        if k.startswith("-"):
            v = copy.deepcopy(v)
            if "out_dir" in k:
                v = os.path.join(ouput_base_dir, v)
                output_dir = v
            if "cfg" in k:
                if not isinstance(v, list):
                    v = [v]
                if append_profile_config:
                    v += [f"./cfg/profiles/{profile}.json"]
            cmd.append(k)
            if isinstance(v, list):
                cmd += v
            else:
                if v is not None:
                    cmd.append(v)
    if output_dir is None:
        cmd.append('--out_dir')
        output_dir = os.path.join(ouput_base_dir, cfg_name)
        cmd.append(output_dir)
    cmd += post_args
    os.makedirs(ouput_base_dir, exist_ok=True)
    if verbose:
        print(f"Run configuration {cfg_name} on {gpu_id} GPU. Data store to a directory '{output_dir}'.\n\tCommand line: {' '.join(cmd)}")
    with open(os.path.join(ouput_base_dir, f"{cfg_name}.log"), 'w') as f:
        try:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=base_path)
        except KeyboardInterrupt:
            pass

        
def process_simulations(base_path: str, ouput_base_dir: str, module_name: str, gpu_list: List, cfgs: dict, additional_args: List = None, verbose:int=1) -> List:
    configs = list()
    profiles = cfgs.get('profiles')
    append_profile_config = cfgs.get('append_profile_config', True)
    pre_args = cfgs.get('pre_args', [])
    post_args = cfgs.get('post_args', []) + (additional_args if additional_args is not None else [])
    for cfg_k, cfg_v in cfgs.get('configurations', dict()).items():
        for profile in profiles:
            elem = {
                'cfg_name': cfg_k,
                'cfg': cfg_v,
                'profile': profile,
                'module_name': module_name,
                'ouput_base_dir': os.path.join(ouput_base_dir, profile),
                'base_path': base_path,
                'gpu_list': gpu_list,
                'append_profile_config': append_profile_config,
                'pre_args': pre_args,
                'post_args': post_args,
                'verbose': verbose
            }
            configs.append(elem)
    num_threads = len(gpu_list)

    if num_threads == 1:
        for config in configs:
            compute(config)
    else:
        with Pool(max_workers=num_threads) as fp:
            try: 
                fp.map(compute, configs)        
            except KeyboardInterrupt:
                print("User aborted.")
                return           
           
def run_eval_script(output_base_dir: str, cfg: str = None, module: str = "src.reco.scripts.eval", additional_args: List = None, verbose:int=True) -> None:
    if cfg is None:
        cur_file_path=os.path.dirname(os.path.abspath(__file__))
        cfg=os.path.join(cur_file_path, os.pardir, "cfg", "eval", "all.json")
    cfgs = load_cfg(cfg)
    gpu_list = config_gpu_list()
    cur_file_path=os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(cur_file_path, os.pardir))
    process_simulations(base_path, output_base_dir, module, gpu_list, cfgs, additional_args=[str(x) for x in additional_args], verbose=verbose)
    

def main():
    args, additional_args = get_args()
    run_eval_script(args.output_base_dir, args.cfg, args.module, additional_args, not args.silent)

if __name__ == "__main__":
    main()