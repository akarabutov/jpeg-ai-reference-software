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
import shutil
import subprocess
import sys
from typing import Any, Dict, Tuple

import GPUtil
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from addict import Dict as DictObj

from src.codec import set_param_recurrent


def load_checkpoint_to_model(model: nn.Module, checkpoint: dict, cur_mk: list, cur_uk: list, *args,
                             **kwargs) -> None:
    mk, uk = model.load_state_dict(checkpoint, *args, **kwargs)
    cur_mk += mk
    cur_uk += uk


def change_tensor_range(x, in_range, out_range):
    if in_range != out_range:
        scale_factor = (out_range[1] - out_range[0]) / (in_range[1] - in_range[0])
        offset = out_range[0] - scale_factor * in_range[0]
        x *= scale_factor
        x += offset


def change_img_range(img, in_ranges, out_ranges):
    if isinstance(img, dict):
        for i, plane in enumerate(['Y', 'U', 'V']):
            change_tensor_range(img[plane], in_ranges[i], out_ranges[i])
    else:
        for i in range(img.shape[1]):
            change_tensor_range(img[:, i], in_ranges[i], out_ranges[i])


def simplified_softmax(x, dim):
    y = F.relu(x) + 1e-12
    return y / y.sum(dim, True)


def safe_clamp(x, min_val, max_val):
    
    type_info = torch.iinfo(x.dtype)
    
    min_val = max(min_val, type_info.min)
    max_val = min(max_val, type_info.max)
    
    return x - x.detach() + x.detach().clamp(min_val, max_val)


def safe_round(x):
    return x - x.detach() + x.detach().round()


def update_dict_recursively(orig_dict: Dict[str, Any], new_dict: Dict[str, Any]) -> Dict[str, Any]:
    for key, val in new_dict.items():
        if isinstance(val, dict):
            tmp = update_dict_recursively(orig_dict.get(key, {}), val)
            orig_dict[key] = tmp
        else:
            orig_dict[key] = new_dict[key]
    return orig_dict


def copy_dict_recursively(orig_dict: Dict[str, Any]) -> Dict[str, Any]:
    ans = dict()
    for k, v in orig_dict.items():
        if isinstance(v, torch.Tensor):
            ans[k] = v.detach().clone()
        elif numpy.isscalar(v):
            ans[k] = v
        else:
            ans[k] = copy_dict_recursively(v)
    return ans


def remove_param_from_dict(d, names):
    if isinstance(names, str):
        names = [names]
    for name in names:
        if name in d:
            del d[name]
    return d


def pop_param_from_dict(d: Dict, name: str, def_val: Any) -> Tuple[Dict, Any]:
    ans = def_val
    if name in d:
        ans = d[name]
        del d[name]
    return d, ans


def param_to_dict(param, val):
    value = val
    try:
        for key in reversed(param.split('.')):
            value = {key: value}
    except:  # noqa: E722
        return {}

    return DictObj(value)


def reverse_range(num):
    return [-(x + 1) for x in range(num)]


def config_gpu_list(kwargs: dict):
    gpu_list = list()

    if kwargs.get('only_cpu', False) or not torch.cuda.is_available():
        import multiprocessing
        num = multiprocessing.cpu_count()
        if kwargs.get('cpu_threads_limit',1) != -1:
            num = min(num, kwargs.get('cpu_threads_limit', 1))
        gpu_strs = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if len(gpu_strs) > 0:
            num = len(gpu_strs.split(','))
        gpu_list = reverse_range(num)

    else:
        # Get list with available GPUs
        gpu_ids = GPUtil.getAvailable(limit=float('inf'))
        if kwargs.get('gpu_greedy', False):
            gpu_ids = [gpu.id for gpu in GPUtil.getGPUs()]
        if kwargs.get('gpu_max', None) is not None:
            gpu_ids = gpu_ids[0:kwargs['gpu_max']]
        if kwargs.get('gpu_ids', None) is not None:
            gpu_ids = [int(gpu_str) for gpu_str in kwargs['gpu_ids'].split(',')]

        # List of enabled GPU
        gpu_strs = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if len(gpu_strs) == 0:
            gpu_list = gpu_ids
        else:
            for gpu_str in gpu_strs.split(','):
                gpu_str = gpu_str.strip()
                if len(gpu_str) > 0 and (int(gpu_str) in gpu_ids):
                    gpu_list.append(int(gpu_str))

    for k in ('gpu_max', 'gpu_greedy', 'gpu_ids' ,'cpu_threads_limit'):
        if k in kwargs:
            del kwargs[k]

    return gpu_list


def setup_dec_device(args, params):
    if args.get('only_cpu', True):
        set_param_recurrent(params, 'target_device', 'cpu')


def start_and_log_str(cmd, f=None, cwd=None, env=None):
    if cwd is None:
        cwd = os.getcwd()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd, env=env)

    while True:
        line = p.stdout.readline()
        if not line:
            break
        else:
            s = line.decode('utf-8').replace('\n', '')
            if f is not None:
                logging_str(s, f)
            else:
                print(s)

    try:
        p.communicate(timeout=100)
    except subprocess.TimeoutExpired:
        p.kill()
        return -1

    return p.returncode


def logging_str(string, fp):
    if isinstance(string, bytes):
        string = string.decode('utf-8')
    print(string)
    fp.write(string + '\n')


def copy_eval_scripts(dst_dir, codec_name: str):
    root_dir = os.path.join(os.getcwd(), 'src', codec_name, 'scripts')
    paths = {'collect_results.py': os.path.join(root_dir, 'collect_results.py')}

    for key in paths:
        src_path, dst_path = paths[key], os.path.join(dst_dir, key)
        shutil.copyfile(src_path, dst_path)


def execute_eval(res_dir, calc_decoder_metrics):
    # Encoder-side metrics
    cmd = [sys.executable, 'collect_results.py']
    subprocess.call(cmd, cwd=res_dir)
    if calc_decoder_metrics:
        cmd = [
            sys.executable, 'collect_results.py', '--enc_log',
            os.path.abspath(os.path.join(res_dir, 'log', 'dec')), '--summary', 'summary_dec.txt'
        ]
        subprocess.call(cmd, cwd=res_dir)
        
def is_int_dtype(tensor):
    return tensor.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]
