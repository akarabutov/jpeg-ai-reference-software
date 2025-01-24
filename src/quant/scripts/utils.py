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

import re
import numpy as np
from .bj_delta import bj_delta
from .quantize_layer import set_int_datatype as set_int_datatype_to_layer_parameters

metrics ={ 'psnr_y': {}, 'ms_ssim_y': {} }

def read_distortion_from_summary(filename, seqname=''):
    with open(filename, 'r') as _file:
        lines = _file.readlines()
    
    for line in lines:
        seq_name_in_summary_regexp = re.compile(r'''VM_(?P<jpegai_number>\d+)_(?P<dataset>VL|TE)_(?P<w>\d+)x(?P<h>\d+)(?P<postfix>.*?)
                                                    _(?P<bitrate>\d+).png\t(?P<bpp>\d+\.\d+)
                                                    \t(?P<ms_ssim_y>\d+\.\d+)\t(\d+\.\d+)\t(?P<psnr_y>\d+\.\d+)''', re.VERBOSE)
        s = seq_name_in_summary_regexp.search(line)
        if s is not None:
            seq_name_params = s.groupdict()
            bitrate = int(seq_name_params['bitrate'])
            
            for metric, metric_per_qp in metrics.items():
                if bitrate not in metric_per_qp:
                    metric_per_qp[bitrate] = 0
                metric_per_qp[bitrate] += float(seq_name_params[metric])      
            
    num_seqs = len(lines) / len( next( iter( metrics.values() ) ) )
            
    for metric_per_qp in metrics.values():
        for key in metric_per_qp:                
            metric_per_qp[key] /= num_seqs
        
    return metrics

def calculate_bd_rate(anchor_float_bpp, test_bpp, dist_metrics):
    
    bd_rate = { key : 0 for key in dist_metrics}
    
    for metric, metric_per_qp in metrics.items():
        metric_per_qp_numpy = np.zeros(len(metric_per_qp))
        for idx, value in enumerate(metric_per_qp.values()):
            metric_per_qp_numpy[idx] = value
                     
        bd_rate[metric] = bj_delta(anchor_float_bpp, metric_per_qp_numpy, test_bpp, metric_per_qp_numpy, mode=1)
    
    avg_bd_rate = 0
    for metric in metrics:
        avg_bd_rate += bd_rate[metric]
        
    avg_bd_rate /= len(bd_rate)

    return avg_bd_rate

def get_layer_by_name(model, layer_name):
    return model.__getattr__(layer_name)

def _set_param_to_all_models(models_for_all_qps, quantized_module, layer_name, param_name, value):
    for model in models_for_all_qps:
        for comp in ['y', 'uv']:
            model[f"{quantized_module}_{comp.upper()}"].__getattr__(layer_name).__setattr__(param_name, value)
            
def set_int_datatype(model, layers_for_quant, weights_bd, bias_bd):
        for layer_name in layers_for_quant:
            layer = model.__getattr__(layer_name)
            set_int_datatype_to_layer_parameters(layer, weights_bd, bias_bd)

def set_name_and_parent_name(models_for_all_qps, quantized_module, layer_name):
    for model in models_for_all_qps:
        for comp in ['y', 'uv']:
            parent_name = f"{quantized_module}_{comp.upper()}"
            model[parent_name].__getattr__(layer_name).parent_name = parent_name
            model[parent_name].__getattr__(layer_name).name = layer_name
