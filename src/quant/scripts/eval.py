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
from copy import deepcopy
import os
import sys
from functools import partial
import torch
import re

from src.codec import get_downloader
from ...codec.scripts import (CodecEval, def_eval_base_parser, def_eval_parser_decorator)
from typing import Tuple, Dict, List
from src.codec.coding_tools.coding_engine import CodingEngine

from .vm_related_code import * #make better interface
from .utils import _set_param_to_all_models, get_layer_by_name, set_int_datatype, set_name_and_parent_name
from .quantize_layer import quantize_layer, post_process_quantized_layer


def def_base_parser(**kwargs):
    this = def_eval_base_parser(**kwargs)
    #this.add_argument('--iter_count')

    return (this)

class QuantEval(CodecEval):
    # ######################################################################################################################
    #  Main methods
    # ######################################################################################################################
    def codec_stream(self,
                     ce_args: Dict,
                     gpu_list: List,
                     params: Dict,
                     set_cmd_for_enc,
                     set_cmd_for_dec,
                     set_cmd_for_cmp,
                     **kwargs):
        pass
        
    def init_codec(self):
        self.process(set_cmd_for_enc=None,
                  set_cmd_for_dec=None,
                  set_cmd_for_cmp=None,
                  encoder_inst=None)
        #There is no dedicated interface for intialization now, so .proccess() with empty codec_stream is used instead
        self.ce.eval()
        self.ce.build_models_recursively()      
        self.ce.load_models_recursively(get_downloader())    

def quantize_hyper_entropy(curr_models, codec = None, dumped_data_dirs = None):
    for models_list_1_qp in curr_models:
        for comp in ['Y', 'UV']:
            models_list_1_qp[f'hyper_entropy_{comp}'].is_quantized = torch.tensor([True], dtype=torch.bool)
            #Actual quantization happen on state_dict()
    return curr_models
            
def quantize_hsd(curr_models, codec, dumped_data_dirs):
    quantized_module = 'hyper_scale_decoder'
    in_bound = 8 - 1 #8 bit
    
    #partial can't bind 2nd argument unfortunatelly
    set_param = lambda models_for_all_qps, layer_name, param_name, value : _set_param_to_all_models(models_for_all_qps, quantized_module, layer_name, param_name, value)   
    layers_for_quant = get_list_of_layers_for_quantization(curr_models[0]) #TODO: Tim: curr_models[0] is not safe
    print('Layers for quantization:', layers_for_quant)
    
    for layer_num, layer_name in enumerate(layers_for_quant):
        set_name_and_parent_name(curr_models, quantized_module, layer_name)
        set_param(curr_models, layer_name, 'in_bound', in_bound)
        
    set_param(curr_models, layers_for_quant[0], 'in_bound', in_bound_for_the_module)
    
    for rate_point_idx, model_list_for_1_rate_point in enumerate(curr_models):   
        models_to_be_quantized = { key : value for key, value in model_list_for_1_rate_point.items() if quantized_module in key }
        
        for model_name, model in models_to_be_quantized.items():
            print(f'===== Quantizing model : {model_name} : rate point {model.id} =====')
            
            anchor_float_bpps = {}
            for dumped_data_dir in dumped_data_dirs:
                avg_bpp = get_average_bpp_for_calibration_set(model, dumped_data_dir, codec)
                
                if avg_bpp == 0.0:
                    print(f'ERROR: there are no data for the model {model_name} id: {model.id}, quantization of this model will be skipped')
                    exit(1)
                print(f'Float anchor bpp for {dumped_data_dir} = {avg_bpp:.4f}')     #TODO: Tim: logging
                
                anchor_float_bpps[dumped_data_dir] = avg_bpp
                
            get_layer_by_name(model, layers_for_quant[0]).in_precision = 0
     
            for layer_num, layer_name in enumerate(layers_for_quant):
                best_loss = float('inf')
                is_last_layer = layer_num == ( len(layers_for_quant) - 1 )
                
                if not is_last_layer:
                    layer = get_layer_by_name( model, layers_for_quant[layer_num + 1] )
                    layer.non_quant_layer_hook_handle = layer.register_forward_pre_hook(non_quantized_layers_forward_pre_hook)  
                
                layer = get_layer_by_name(model, layer_name)
                shift_lower_bound = max(0, output_precision_for_the_module - layer.in_precision) if is_last_layer else 0
                
                quantize_layer(layer, args.weights_bd, args.bias_bd, shift_lower_bound = shift_lower_bound)
                max_precision = layer.per_channel_shifts.min() + layer.in_precision #it is needed to avoid negative shifts
                assert(not is_last_layer or output_precision_for_the_module <= max_precision)
                
                search_range = range(max_precision, args.precision_min - 1, -1) if not is_last_layer else [output_precision_for_the_module]
                
                for precision in search_range:
                    tested_model = deepcopy(model)
                    curr_layer = get_layer_by_name(tested_model, layer_name)
                    curr_layer.out_precision = precision
                    post_process_quantized_layer( curr_layer, args.bias_bd ) #can be done only with knowledge about out_precision
                    
                    if not is_last_layer:
                        get_layer_by_name( tested_model, layers_for_quant[layer_num + 1] ).in_precision = precision
                    
                    test_loss = get_scaled_average_bpp_for_calibration_set(tested_model, dumped_data_dirs, codec, anchor_float_bpps)
                    
                    print(f'layer: {layer_name}: precision = {precision} \t test loss = {test_loss:.4f}, best loss: {best_loss:.4f}')
                        
                    if test_loss < best_loss:
                        bpp_change = ( test_loss - 1 ) * 100
                        best_loss = test_loss
                        best_model = tested_model
                        print(f'===>    better bpp is found ===> bpp_change: {bpp_change:.4f} % )')
                        
                model = best_model
            
            set_int_datatype(model, layers_for_quant, args.weights_bd, args.bias_bd)
            curr_models[rate_point_idx][model_name] = model
    
    return curr_models

def non_quantized_layers_forward_pre_hook(self, input):
    x = input[0]
    x.clamp_(-self.max_signal_value, self.max_signal_value)
    x = x.to(torch.float32) / (2 ** self.in_precision)
    return x

def quantize_layer_in_all_models(models, quantized_module, layer_name, weights_bd, bias_bd):
    for model in models:
        for comp in ['y', 'uv']:
            layer = model[f"{quantized_module}_{comp.upper()}"].__getattr__(layer_name)
            quantize_layer(layer, weights_bd, bias_bd)

def get_average_bpp_for_calibration_set(model, dumped_data_dir, codec):
    num_bpps = len(codec.target_bpps)
    
    model_data_dir = os.path.join(dumped_data_dir, 'model_' + ('y' if model.isLuma else 'uv') )
    z_hat_dir      = os.path.join(model_data_dir, 'z_hat/')
    res_y_hat_dir  = os.path.join(model_data_dir, 'residual_quant/')
    
    total_bits = 0
    seqs = os.listdir(z_hat_dir)
    num_seqs = len(seqs) / num_bpps
    
    for current_seq in seqs:
        img_name_regexp = re.compile('(?P<jpegai_number>\d+)_(TE|VL)_(?P<w>\d+)x(?P<h>\d+).*?.png_(?P<bitrate>\d+)')
        p = img_name_regexp.search(current_seq)
        seq_params = {k: v for k, v in p.groupdict().items() if v is not None}   
        
        w, h = int(seq_params['w']), int(seq_params['h'])
        bitrate = int( seq_params['bitrate'] )
        model_id = get_model_id_for_bitrate(bitrate, codec)
        
        if model_id == model.id:
            z_hat     = torch.load(os.path.join(z_hat_dir,     current_seq))
            res_y_hat = torch.load(os.path.join(res_y_hat_dir, current_seq))
            num_bits = encode(codec, z_hat, res_y_hat, model, h, w, bitrate)   
            bpp = num_bits / (h * w)
            total_bits += bpp
        
    avg_bpp = total_bits / num_seqs
        
    return avg_bpp

def get_scaled_average_bpp_for_calibration_set(model, dumped_data_dirs, codec, scaling_factors_per_dir):
    
    scaled_bpp = 0
    
    for dumped_data_dir in dumped_data_dirs:
        bpp = get_average_bpp_for_calibration_set(model, dumped_data_dir, codec)
        scaled_bpp += bpp / scaling_factors_per_dir[dumped_data_dir]
        
    return scaled_bpp / len(dumped_data_dirs)
        
def main(args, codec: CodingEngine):
    curr_models = init_models(codec)
    
    quant_params_search_functions = {'hyper_scale_decoder': quantize_hsd, 'hyper_entropy': quantize_hyper_entropy}
    modules_to_quant = ['hyper_entropy', 'hyper_scale_decoder']
    
    dumped_data_dirs = [ os.path.join(args.dumped_data_dir, subdir) for subdir in args.dumped_data_subdirs]
    
    for module in modules_to_quant:
        curr_models = quant_params_search_functions[module](curr_models, codec, dumped_data_dirs)
    
    save_quantized_models(args.model_dir, args.quantized_models_dir, curr_models, modules_to_quant, codec)
        

if __name__ == '__main__':
    """main method
    """
    base_parser = def_base_parser() #tools_quant.json
    parser_decorator = def_eval_parser_decorator(argparse.ArgumentParser())
    coder = QuantEval('quant', base_parser, parser_decorator)
    
    print("Initializing VM")
    
    coder.init_codec()
    
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument('--dumped_data_dir',     type=str, default='data/dump',
                        help='Directory to read summary, zhat and yhat data. --dumped_data_subdirs should be additionally set up')
    parser.add_argument('--dumped_data_subdirs', type=str, nargs='+', default=['bop', 'hop'], help='directory to read summary, zhat and yhat data')
    parser.add_argument('--model_dir',    type=str, default='models/VM_common', help='float models path')
    # log and debug
    parser.add_argument('--debug',        type=int, default=1, help='')
    # search quantization params
    parser.add_argument('--weights_bd',    type=int, default=8,  help='weights bit depth')
    parser.add_argument('--bias_bd',       type=int, default=30, help='bit depth of biases')
    parser.add_argument('--precision_min', type=int, default=0,  help='smallest precision value to search')
    # results saving
    parser.add_argument('--quantized_models_dir',type=str, default='./models/VM_common_int', help='result quantized models are saved')

    args, _ = parser.parse_known_args()

    main(args, coder.ce)
        
    print("All set.")
    
