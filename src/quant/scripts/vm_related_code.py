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

import torch
import os

from copy import deepcopy
from src.codec.coding_tools.coding_engine import CodingEngine
#from src.codec.bitstream_structure import BitstreamStructure
from src.codec.entropy_coding import ECModule, create_lh_ecmodule

in_bound_for_the_module = 7 # zhat is integer (precision for the first layer is fixed to 0) in_bound depends on z_offset, which is fixed.
output_precision_for_the_module = 7

def init_models(codec):
    models_list_all_qps = []
    
    for id, tool in enumerate( codec.model.get_tool().tools ):
        models_list_1_qp = {}
        
        for comp in ['Y', 'UV']: 
            sep_chan_tool = tool.model_y if comp == 'Y' else tool.model_uv  
            common_modules = sep_chan_tool.common_modules
            hyper_scale_decoder = deepcopy(common_modules.hyper_scale_decoder).eval() #TODO: use model_X. ... addressing
            hyper_scale_decoder.isLuma  = comp == 'Y'
            hyper_scale_decoder.id = id
            models_list_1_qp[f'hyper_scale_decoder_{comp}'] = hyper_scale_decoder
            models_list_1_qp[f'hyper_entropy_{comp}']       = deepcopy(common_modules.hyper_entropy).eval()
            
        models_list_all_qps.append(models_list_1_qp)
   
    return models_list_all_qps

def get_model_id_for_bitrate(bitrate, codec):
    bpp_idx  = codec.model.bitrate_matcher.default_target_rates.index(bitrate)
    model_id = codec.model.bitrate_matcher.default_models[bpp_idx]
    return model_id
    
def encode(codec: CodingEngine, z_hat, res_y_hat, hyper_scale_decoder, h, w, bitrate):
    
    #z_hat = torch.clamp(z_hat, -z_range, z_range - z_offset - 1) #TODO: Tim: check whether it's clipped before the dumping
    
    main_model = codec.model.get_tool()
    
    cached_target_bpps = codec.target_bpps
    codec.target_bpps = [ bitrate ]
    codec.model.bitrate_matcher.process(main_model, None)
    
    tool = main_model.get_active_tool()
    
    if hyper_scale_decoder.isLuma:
        sep_chan_tool = tool.model_y
        sep_chan_tool.beta_displacement_log = main_model.beta_displacement_log_Y #TODO: Tim: dog-nail
    else:
        sep_chan_tool = tool.model_uv
        [h, w] = tool.calc_downsampled_shape(h, w)
        sep_chan_tool.beta_displacement_log = main_model.beta_displacement_log_UV
    
    scale_log = hyper_scale_decoder(z_hat, h, w)
    common_modules = sep_chan_tool.common_modules
    scale_log = common_modules.quantizer.quantize_scale(scale_log, None, incl_list=['gain_unit'])

    #bs = BitstreamStructure(codec.EC, coder_direction=0) #TODO: Tim: dog-nail
    #bs.encode_init() #TODO: use values from params
    #AE = 
    #ec = ECModule(bs, False, profilers=codec.get_profilers())
    ec = create_lh_ecmodule()

    common_modules._ac_encode_z(ec, z_hat)
    masks = torch.ones_like(scale_log).bool()
    common_modules._ac_encode_y(ec, res_y_hat, scale_log, masks)
    
    num_bits = ec.get_total_bits().item()    
    #ec.bs.encode_term(None)
    
    codec.target_bpps = cached_target_bpps

    return num_bits

def get_list_of_layers_for_quantization(models):
    
    quantized_layers = []
    
    modules_dict = dict( models['hyper_scale_decoder_Y'].named_modules() )
    
    for name, layer in modules_dict.items():
        if isinstance(layer, torch.nn.ConvTranspose2d) or isinstance(layer, torch.nn.Conv2d):
            quantized_layers.append(name)
            
    return quantized_layers

def save_quantized_models(models_dir, quantized_models_dir, quantized_models, quantized_modules, codec):
    
    print(f"Saving quantized models to {quantized_models_dir}...")
    
    components = ['Y', 'UV']
    
    from pathlib import Path
    Path(quantized_models_dir).mkdir(parents=True, exist_ok=True)
    
    for model_idx, quantized_model in enumerate(quantized_models):
        model_for_1_rate_point = codec.model.get_tool().tools[model_idx]
        
        for component_model in model_for_1_rate_point.models_list:
            ckpt_files = component_model.common_modules.get_ckpt_names()
            assert len(ckpt_files) == 1
            ckpt_file = ckpt_files[0]
            
            non_quantized_model_path = os.path.join(models_dir,           ckpt_file)
            quantized_model_path     = os.path.join(quantized_models_dir, ckpt_file)
            checkpoint = torch.load(non_quantized_model_path)
        
            for module in quantized_modules:
                assert( component_model.name in ['model_y', 'model_uv'] )
                module_full_name = f"{module}_{'Y' if component_model.name == 'model_y' else 'UV'}"
                checkpoint.update( quantized_model[module_full_name].state_dict( prefix = module+'.' ) )
                
            torch.save(checkpoint, quantized_model_path)
        
def extract_sub_state_dict(dict, parent_key):
    return { '.'.join(key.split('.')[1:]) : value for key, value in dict.items() if parent_key == key.split('.')[0] }
                
    