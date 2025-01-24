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
import torch
file_abs_path = os.path.abspath(__file__)
prog_dir = os.path.dirname(os.path.dirname(os.path.dirname(file_abs_path)))
sys.path.append(os.path.join(prog_dir))

sys.path.append(file_abs_path)
from channel_list import sorted_channel_ids_dict

def reorder_weight_and_bias(ckpt_name, model_key, out_ckpt_name, is_input=False):
    state_dict = torch.load(ckpt_name)
    gain_vector_names = ['vr_vec']
    mcm_input_layer_names = [
            #'context.MCM.0.fusion_pred_net.conv1', 
            #'context.MCM.1.fusion_pred_net.conv1',
            #'context.MCM.2.fusion_pred_net.conv1', 
            #'context.MCM.3.fusion_pred_net.conv1', 
            'context.MCM.1.conv.0', # [cout, cin, ks, ks]
            'context.MCM.2.conv.0', 
            'context.MCM.3.conv.0'
        ]
    input_layer_names = ['hyper_encoder.conv1']
    hyper_scale_decoder_out_layer = ['hyper_scale_decoder.pointwise']
    #hyper_decoder_out_layer = ['hyper_decoder.conv4']
    hyper_decoder_out_layer = []
    mcm_output_layer_names = [
            'context.MCM.0.fusion_pred_net.conv3',
            'context.MCM.1.fusion_pred_net.conv3', 
            'context.MCM.2.fusion_pred_net.conv3', 
            'context.MCM.3.fusion_pred_net.conv3', 
            #'context.MCM.1.conv.1', 
            #'context.MCM.2.conv.1', 
            #'context.MCM.3.conv.1'
        ]
    for layer_name in gain_vector_names:
        num_latent_channels_luma = 160
        param_name = layer_name + ".c"
        param = state_dict[param_name]
        param = param[sorted_channel_ids_dict[model_key], :]
        state_dict[param_name] = param

    for layer_name in mcm_output_layer_names:
        num_latent_channels_luma = 160
        param_name = layer_name + ".weight"
        param = state_dict[param_name]
        param = param[sorted_channel_ids_dict[model_key], :, :, :]
        state_dict[param_name] = param
        
        bias_name = layer_name + ".bias"
        if bias_name in state_dict:
            bias = state_dict[bias_name]
            bias = bias[sorted_channel_ids_dict[model_key]]
            state_dict[bias_name] = bias

    for layer_name in mcm_input_layer_names:
        num_latent_channels_luma = 160
        param_name = layer_name + ".weight"
        if param_name in state_dict:
            param = state_dict[param_name]
            num_input_slices = int(param.shape[1] / num_latent_channels_luma)
            for id_slice in range(num_input_slices):
                new_param_id0 = id_slice * num_latent_channels_luma
                new_param_id1 = (id_slice + 1) * num_latent_channels_luma
                old_param_ids = [item + new_param_id0 for item in sorted_channel_ids_dict[model_key]]
                param[:, new_param_id0: new_param_id1, :, :] = param[:, old_param_ids, :, :]
            state_dict[param_name] = param
                #old_param_id0 = id_slice * num_latent_channels_luma + sorted_channel_ids_dict[model_key]
                #old_param_id1 = (id_slice + 1)* num_latent_channels_luma + sorted_channel_ids_dict[model_key]

    for layer_name in input_layer_names: # input layer, HE
        state_dict[layer_name + ".weight"] = state_dict[layer_name + ".weight"][:, sorted_channel_ids_dict[model_key], :, :]

    for layer_name in hyper_scale_decoder_out_layer: # HSD
        if layer_name + ".weight" in state_dict:
            new_tensor = state_dict[layer_name + ".weight"].clone()
            for idx in range(int(state_dict[layer_name + ".weight"].shape[0]/16)):
                old_id = sorted_channel_ids_dict[model_key][idx]
                new_tensor[idx*16:16*idx+16, :, :, :] = state_dict[layer_name + ".weight"][old_id*16:old_id*16+16, :, :, :]
            state_dict[layer_name + ".weight"] = new_tensor
        if layer_name + ".bias" in state_dict:
            new_tensor = state_dict[layer_name + ".bias"].clone()
            for idx in range(int(state_dict[layer_name + ".bias"].shape[0]/16)):
                old_id = sorted_channel_ids_dict[model_key][idx]
                new_tensor[idx*16:16*idx+16] = state_dict[layer_name + ".bias"][old_id*16:old_id*16+16]
            state_dict[layer_name + ".bias"] = new_tensor
        if layer_name + ".quant_params" in state_dict:
            new_tensor = state_dict[layer_name + ".quant_params"].per_channel_shifts.clone()
            for idx in range(int(state_dict[layer_name + ".quant_params"].per_channel_shifts.shape[1]/16)):
                old_id = sorted_channel_ids_dict[model_key][idx]
                new_tensor[:, idx*16:16*idx+16, :, :] = state_dict[layer_name + ".quant_params"].per_channel_shifts[:, old_id*16:old_id*16+16, :, :]
            state_dict[layer_name + ".quant_params"].per_channel_shifts = new_tensor

    for layer_name in hyper_decoder_out_layer:
        new_tensor = state_dict[layer_name + ".weight"].clone()
        for idx in range(4):
            num_channels = int(state_dict[layer_name + ".weight"].shape[0]//4)
            for channel_id in range(num_channels):
                new_tensor_chid = idx * num_channels + channel_id
                old_tensor_chid = sorted_channel_ids_dict[model_key][channel_id] + idx * num_channels
                new_tensor[new_tensor_chid, :, :, :] = state_dict[layer_name + ".weight"][old_tensor_chid, :, :, :]
        state_dict[layer_name + ".weight"] = new_tensor
 
        if layer_name + ".bias" in state_dict:
            new_tensor = state_dict[layer_name + ".bias"].clone()
            for idx in range(4):
                num_channels = int(state_dict[layer_name + ".bias"].shape[0]//4)
                for channel_id in range(num_channels):
                    new_tensor_chid = idx * num_channels + channel_id
                    old_tensor_chid = sorted_channel_ids_dict[model_key][channel_id] + idx * num_channels
                    new_tensor[new_tensor_chid] = state_dict[layer_name + ".bias"][old_tensor_chid]
            state_dict[layer_name + ".bias"] = new_tensor
 
    torch.save(state_dict, out_ckpt_name)
    
ckpt_names = ["Y_0.002.pth", "Y_0.012.pth", "Y_0.075.pth", "Y_0.5.pth"]
model_keys = ["Y_0.002", "Y_0.012", "Y_0.075", "Y_0.5"]
out_ckpt_names = ["Y_0.002.pth", "Y_0.012.pth", "Y_0.075.pth", "Y_0.5.pth"]
for idx in range(len(ckpt_names)):
    ckpt_name = ckpt_names[idx]
    model_key = model_keys[idx]
    #model_key = "Y_0.075"
    out_ckpt_name = out_ckpt_names[idx]
    reorder_weight_and_bias(ckpt_name, model_key, out_ckpt_name, is_input=False)

def reorder_weight_and_bias_uv(ckpt_name, model_key, out_ckpt_name, is_input=False):
    state_dict = torch.load(ckpt_name)
    gain_vector_names = ['vr_vec']
    for layer_name in gain_vector_names:
        num_latent_channels_luma = 96
        param_name = layer_name + ".c"
        param = state_dict[param_name]
        param = param[sorted_channel_ids_dict[model_key], :]
        state_dict[param_name] = param

    hyper_scale_decoder_out_layer = ['hyper_scale_decoder.pointwise']
    for layer_name in hyper_scale_decoder_out_layer: # HSD
        if layer_name + ".weight" in state_dict:
            new_tensor = state_dict[layer_name + ".weight"].clone()
            for idx in range(int(state_dict[layer_name + ".weight"].shape[0]/16)):
                old_id = sorted_channel_ids_dict[model_key][idx]
                new_tensor[idx*16:16*idx+16, :, :, :] = state_dict[layer_name + ".weight"][old_id*16:old_id*16+16, :, :, :]
            state_dict[layer_name + ".weight"] = new_tensor
        if layer_name + ".bias" in state_dict:
            new_tensor = state_dict[layer_name + ".bias"].clone()
            for idx in range(int(state_dict[layer_name + ".bias"].shape[0]/16)):
                old_id = sorted_channel_ids_dict[model_key][idx]
                new_tensor[idx*16:16*idx+16] = state_dict[layer_name + ".bias"][old_id*16:old_id*16+16]
            state_dict[layer_name + ".bias"] = new_tensor
        if layer_name + ".quant_params" in state_dict:
            new_tensor = state_dict[layer_name + ".quant_params"].per_channel_shifts.clone()
            for idx in range(int(state_dict[layer_name + ".quant_params"].per_channel_shifts.shape[1]/16)):
                old_id = sorted_channel_ids_dict[model_key][idx]
                new_tensor[:, idx*16:16*idx+16, :, :] = state_dict[layer_name + ".quant_params"].per_channel_shifts[:, old_id*16:old_id*16+16, :, :]
            state_dict[layer_name + ".quant_params"].per_channel_shifts = new_tensor

    input_layer_names = ['hyper_encoder.conv1']
    for layer_name in input_layer_names: # input layer, HE
        state_dict[layer_name + ".weight"] = state_dict[layer_name + ".weight"][:, sorted_channel_ids_dict[model_key], :, :]

    hyper_decoder_out_layer = ['hyper_decoder.conv4']
    for layer_name in hyper_decoder_out_layer:
        new_tensor = state_dict[layer_name + ".weight"].clone()
        for idx in range(4):
            num_channels = int(state_dict[layer_name + ".weight"].shape[0]//4)
            for channel_id in range(num_channels):
                new_tensor_chid = idx * num_channels + channel_id
                old_tensor_chid = sorted_channel_ids_dict[model_key][channel_id] + idx * num_channels
                new_tensor[new_tensor_chid, :, :, :] = state_dict[layer_name + ".weight"][old_tensor_chid, :, :, :]
        state_dict[layer_name + ".weight"] = new_tensor
 
        if layer_name + ".bias" in state_dict:
            new_tensor = state_dict[layer_name + ".bias"].clone()
            for idx in range(4):
                num_channels = int(state_dict[layer_name + ".bias"].shape[0]//4)
                for channel_id in range(num_channels):
                    new_tensor_chid = idx * num_channels + channel_id
                    old_tensor_chid = sorted_channel_ids_dict[model_key][channel_id] + idx * num_channels
                    new_tensor[new_tensor_chid] = state_dict[layer_name + ".bias"][old_tensor_chid]
            state_dict[layer_name + ".bias"] = new_tensor
    torch.save(state_dict, out_ckpt_name)

ckpt_names = ["UV_0.002.pth", "UV_0.012.pth", "UV_0.075.pth", "UV_0.5.pth"]
model_keys = ["UV_0.002", "UV_0.012", "UV_0.075", "UV_0.5"]
out_ckpt_names = ["UV_0.002.pth", "UV_0.012.pth", "UV_0.075.pth", "UV_0.5.pth"]
for idx in range(len(ckpt_names)):
    ckpt_name = ckpt_names[idx]
    model_key = model_keys[idx]
    out_ckpt_name = out_ckpt_names[idx]
    reorder_weight_and_bias_uv(ckpt_name, model_key, out_ckpt_name, is_input=False)
