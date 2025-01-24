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
sys.path.append(file_abs_path)
from channel_list import sorted_channel_ids_dict

def reorder_weight_and_bias(ckpt_name, model_key, layer_name, out_ckpt_name, is_input=False):
    state_dict = torch.load(ckpt_name)
    if is_input:
        if layer_name + ".weight" in state_dict:
            state_dict[layer_name + ".weight"] = state_dict[layer_name + ".weight"][:, sorted_channel_ids_dict[model_key], :, :]
    else:
        if layer_name + ".weight" in state_dict:
            state_dict[layer_name + ".weight"] = state_dict[layer_name + ".weight"][sorted_channel_ids_dict[model_key], :, :, :]
        #if layer_name + ".bias" in state_dict:
        #    state_dict[layer_name + ".bias"] = state_dict[layer_name + ".bias"]
    clip_key = 'clip_thres'
    last_layer_key = 'E5H'
    key_list = ['MaxList', 'MinList']
    if clip_key in state_dict:
        if last_layer_key in state_dict[clip_key]:
            for key in key_list:
                if key in state_dict[clip_key][last_layer_key]:
                    clip_val = state_dict[clip_key][last_layer_key][key].copy()
                    for idx in range(len(sorted_channel_ids_dict[model_key])):
                        state_dict[clip_key][last_layer_key][key][idx] = clip_val[sorted_channel_ids_dict[model_key][idx]]
    torch.save(state_dict, out_ckpt_name)
    
ckpt_names = ["encoder_Y_0.002.pth", "encoder_Y_0.012.pth", "encoder_Y_0.075.pth", "encoder_Y_0.5.pth"]
model_keys = ["Y_0.002", "Y_0.012", "Y_0.075", "Y_0.5"]
out_ckpt_names = ["encoder_Y_0.002.pth", "encoder_Y_0.012.pth", "encoder_Y_0.075.pth", "encoder_Y_0.5.pth"]

ckpt_names += ["encoder_UV_0.002.pth", "encoder_UV_0.012.pth", "encoder_UV_0.075.pth", "encoder_UV_0.5.pth"]
model_keys += ["UV_0.002", "UV_0.012", "UV_0.075", "UV_0.5"]
out_ckpt_names += ["encoder_UV_0.002.pth", "encoder_UV_0.012.pth", "encoder_UV_0.075.pth", "encoder_UV_0.5.pth"]

layer_name = "conv5"

for idx in range(len(ckpt_names)):
    ckpt_name = ckpt_names[idx]
    model_key = model_keys[idx]
    #model_key = "Y_0.075"
    out_ckpt_name = out_ckpt_names[idx]
    reorder_weight_and_bias(ckpt_name, model_key, layer_name, out_ckpt_name, is_input=False)
