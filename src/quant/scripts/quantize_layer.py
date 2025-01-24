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
import torch.nn as nn

def quantize_layer(layer, weights_bd, bias_bd, shift_lower_bound):
    
    search_optimal_precision = True if weights_bd <= 16 else False
    
    per_channel_shifts = _prepare_per_channel_shifts(layer, search_optimal_precision, weights_bd, shift_lower_bound)
    
    layer.per_channel_shifts = per_channel_shifts
    
    q_weight, q_bias = _quantize_layer_parameters(layer.weight, layer.bias, weights_bd, bias_bd, layer.in_precision, layer._get_per_channel_shifts_for_weights)
    
    if q_bias is None:
        device = layer._get_weights_of_ith_channel(0).device
        layer.bias = torch.nn.Parameter( torch.zeros( layer.out_channels, device = device) ) #bias is always need for quantized convs in current design
        q_bias = layer.bias.data
    
    q_weight, q_bias = _set_int_datatype(q_weight, weights_bd, q_bias, bias_bd)
    
    layer.is_quantized.fill_(True)
    
    layer_state_dict = layer.state_dict()
    layer_state_dict['weight'] = q_weight
    layer_state_dict['bias']   = q_bias
    
    layer.load_state_dict(layer_state_dict, strict = False)
    
    if hasattr(layer, 'non_quant_layer_hook_handle'):
        layer.non_quant_layer_hook_handle.remove()

def _prepare_per_channel_shifts(layer, search_optimal_precision, weights_bd, shift_lower_bound):
    num_out_cahnnels = layer.out_channels
    device = layer._get_weights_of_ith_channel(0).device

    conv_ch_negative_sum     = torch.zeros(num_out_cahnnels, device=device)
    conv_ch_positive_sum     = torch.zeros(num_out_cahnnels, device=device)
    conv_ch_abs_positive_sum = torch.zeros(num_out_cahnnels, device=device)
    additional_precision     = torch.zeros(num_out_cahnnels, device=device)
    per_channel_shifts       = torch.zeros(num_out_cahnnels, device=device, dtype=torch.int8)

    for i in range(num_out_cahnnels):
        conv_ch = layer._get_weights_of_ith_channel(i)
        conv_ch_negative = conv_ch[conv_ch < 0]
        conv_ch_positive = conv_ch[conv_ch > 0]
        conv_ch_negative_sum[i] = conv_ch_negative.sum()
        conv_ch_positive_sum[i] = conv_ch_positive.sum()
        conv_ch_abs_positive_sum[i] = conv_ch_positive.sum() - conv_ch_negative.sum()
        additional_precision[i] = conv_ch_abs_positive_sum[i].log2().ceil()
        per_channel_shifts[i] = layer.acc_bd_minus_1 - layer.in_bound - additional_precision[i]

    if layer.bias is not None: #TODO: Tim: this place can be optimized
        bias_maximum_allowed_shift = ( layer.acc_bd_minus_1 - torch.max(layer.bias.data.abs().log2().ceil(), torch.zeros(1, device=device)) ).to(dtype=torch.int8)
        per_channel_shifts = torch.min(per_channel_shifts, bias_maximum_allowed_shift)
        per_channel_shifts -= 1

    if per_channel_shifts.min() < 0:
        print('WARNING:Quantized_conv:negative_shift') #TODO: assert?
        
    if search_optimal_precision:
        per_channel_shifts = _search_weights_precision( per_channel_shifts, layer._get_weights_of_ith_channel, weights_bd, shift_lower_bound )

    return per_channel_shifts

def _search_weights_precision( max_allowed_shifts, _get_weights_of_ith_channel, weights_bd, shift_lower_bound ):
    
    out_channels = max_allowed_shifts.numel()
    max_weights_value = 2 ** (weights_bd - 1) - 1
    min_weights_value = -max_weights_value - 1
    precision_opt = 0
    opt_per_channel_shifts = torch.zeros_like(max_allowed_shifts)
    
    for i in range(out_channels):
        best_loss = float('inf')
        float_weights  = _get_weights_of_ith_channel(i).clone()
        max_allowed_shift = max_allowed_shifts[i]
        
        for precision in range(shift_lower_bound, max_allowed_shift + 1, 1):
            quantized_weights = float_weights * (2 ** precision)
            quantized_weights.round_()
            quantized_weights.clamp_(min_weights_value, max_weights_value)
            quantized_weights = quantized_weights / 2 ** precision
     
            loss = nn.functional.mse_loss(float_weights, quantized_weights)
            if loss < best_loss:
                best_loss = loss
                precision_opt = precision
        
        opt_per_channel_shifts[i] = precision_opt
        
    return opt_per_channel_shifts

def _quantize_layer_parameters(weight, bias, weights_bd, bias_bd, input_precision, _get_per_channel_shifts_for_weights):
    quantized_weight = torch.bitwise_left_shift(weight, _get_per_channel_shifts_for_weights())
    quantized_weight.round_()

    weights_max_value = ( 1 << ( weights_bd - 1 ) ) - 1
    quantized_weight = torch.clamp( quantized_weight, -weights_max_value - 1, weights_max_value )

    if bias is not None:
        quantized_bias = torch.bitwise_left_shift(bias, _get_per_channel_shifts_for_weights().view(-1) + input_precision) #TODO: Tim make it INT32 in search as well in future
        quantized_bias.round_()
        bias_max_value = (1 << ( bias_bd - 1 ))
        quantized_bias = torch.clamp( quantized_bias, -bias_max_value, bias_max_value - 1 )
    else:
        quantized_bias = None

    return quantized_weight, quantized_bias

def _set_int_datatype(weight, weights_bd, bias, bias_bd):
    int_weight = weight.to(dtype=torch.int8 if weights_bd <= 8 else torch.int16)
    
    if bias is not None:
        int_bias = bias.to(dtype=torch.int8 if bias_bd <= 8 else torch.int32)
    else:
        int_bias = bias
        
    return int_weight, int_bias

def set_int_datatype(layer, weights_bd, bias_bd):
    layer.weight.data, bias = _set_int_datatype(layer.weight, weights_bd, layer.bias, bias_bd)
    
    if bias is not None:
        layer.bias.data = bias
        
def post_process_quantized_layer(layer, bias_bd):
    
    layer.per_channel_shifts += (-layer.out_precision + layer.in_precision) 
    
    assert( (layer.per_channel_shifts >= 0).all() )
    
    offset = 1 << ( layer.per_channel_shifts.to(dtype=torch.int32) - 1 )
    
    bias_max_value = (1 << ( bias_bd - 1 )) 
    
    bias = ( layer.bias.data.to(dtype=layer.pipeline_datatype) + offset.to(dtype=layer.pipeline_datatype).view(layer.out_channels) )
    clamped_bias = bias.clamp(-bias_max_value , bias_max_value - 1)
    
    if (bias != clamped_bias).any():
        print("WARNING: bias is clamped, quantization loss can increase")
        
    layer.bias.data = clamped_bias
