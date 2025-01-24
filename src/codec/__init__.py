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

import json
import os
import re
import shutil


def get_correct_bit_name(codec_name, seq_name, bpp):
    # Exclude resolution from the name
    r = re.compile('(?P<name>.*)_(\d+)x(\d+)')
    n = r.search(seq_name)
    if n is not None:
        seq_name = n.group('name')
    return f'{codec_name}_{seq_name}_{bpp:03d}'


def get_correct_rec_name(codec_name, seq_name, bits, fmt, bpp):
    r = re.compile('(?P<name>.*)_(?P<w>\d+)x(?P<h>\d+)')
    n = r.search(seq_name)
    if n is not None:
        seq_name = n.group('name')    
        w = n.group('w')
        h = n.group('h')
    return f'{codec_name}_{seq_name}_{w}x{h}_{bits}bit_{fmt}_{bpp:03d}'


def get_cfg_def_dir():
    return os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'cfg')


def CTC_get_default_fn():
    info_path = os.path.join(get_cfg_def_dir(), 'info.json')
    ans = [os.path.join(get_cfg_def_dir(), 'tools-off.json'), os.path.join(get_cfg_def_dir(), 'profiles', 'high.json')]
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            a = json.load(f)
        if 'config' in a:
            cfgs = a['config']
            ans = list()
            for cfg in cfgs:
                ans.append(os.path.join(get_cfg_def_dir(), cfg))
    return ans


def get_pipeline_desc_paths():
    info_path = os.path.join(get_cfg_def_dir(), 'info.json')
    ans = [os.path.join(get_cfg_def_dir(), 'pipeline.json')]
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            a = json.load(f)
        if 'pipeline' in a:
            cfgs = a.get('pipeline', list())
            ans = list()
            if not isinstance(cfgs, list):
                cfgs = [cfgs]
            for cfg in cfgs:
                ans.append(os.path.join(get_cfg_def_dir(), cfg))
    return ans

def get_codec_name():
    ver_path = os.path.join(get_cfg_def_dir(), 'info.json')
    if os.path.exists(ver_path):
        with open(ver_path, 'r') as f:
            a = json.load(f)
        return a['codec_name']
    else:
        return 'unknown'


def get_codec_version():
    ver_path = os.path.join(get_cfg_def_dir(), 'info.json')
    if os.path.exists(ver_path):
        with open(ver_path, 'r') as f:
            a = json.load(f)
        return a['version']
    else:
        return 'unknown'

def get_profiles_dir():
    return os.path.join(get_cfg_def_dir(), "profiles")


def get_downloader(models_dir_name='models', *args, **kwargs):
    from src.codec.utils import Downloader
    path = models_dir_name if os.path.isabs(models_dir_name) else os.path.join(os.getcwd(), models_dir_name)
    return Downloader(path, *args, **kwargs)


def set_param_recurrent(params, param_name, param_value):
    for p in params:
        if isinstance(params[p], dict):
            params[p] = set_param_recurrent(params[p], param_name, param_value)
        if p == param_name:
            params[p] = param_value
    return params


def create_dir_structure(o_path,
                         dirs=['bit', 'ori', 'rec'],
                         overwrite=False,
                         ignore_existance=False):
    out_dirs = []
    for d in dirs:
        p = os.path.join(o_path, d)
        out_dirs.append(p)
        if overwrite and os.path.exists(p):
            shutil.rmtree(p)
        os.makedirs(p, exist_ok=ignore_existance)
    return out_dirs

