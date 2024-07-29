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

from src.codec.common.utils import safe_clamp, is_int_dtype

class ConvIntegerBase(object):
    float_mantissa = 23  # float precision mantissa
    double_mantissa = 52  # double precision mantissa

    def __init__(self, num_output_channels):
        self.acc_bd_minus_1    = 31
        in_bound               = 7
        self.max_signal_value  = (2**in_bound) - 1
        
        self.register_buffer('is_quantized', torch.tensor([False]))
        self.register_buffer('per_channel_shifts', torch.zeros( (1, num_output_channels, 1, 1), dtype=torch.int8) )
        
    def forward(self, x_in):
        
        if not self.is_quantized:
            if is_int_dtype(x_in):
                x_in = x_in.to(self.weight.dtype)
            
            return self.base_conv_function(self, x_in)
        
        x_in = safe_clamp(x_in, -self.max_signal_value - 1, self.max_signal_value)
        x_in = x_in.to(dtype=self.pipeline_datatype)
        
        res = self.base_conv_function(self, x_in)
        res = res.to(dtype = torch.int32)
        
        assert(res.dtype == torch.int32 and self.per_channel_shifts.dtype == torch.int8)

        #res = torch.bitwise_right_shift(res, self.per_channel_shifts)
        #res = res // 2**self.per_channel_shifts
        res = res >> self.per_channel_shifts

        return res
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.pipeline_datatype = self._decide_pipeline_datatype()
        
        self.weight.data = self.weight.data.to(dtype=self.pipeline_datatype)
        if self.bias is not None:
            self.bias.requires_grad = False # TODO: is it a problem for training script?
            self.bias.data = self.bias.data.to(dtype=self.pipeline_datatype)
            
        self.per_channel_shifts = self.per_channel_shifts.view(1, self.out_channels, 1, 1)
            
    def _decide_pipeline_datatype(self):
        if self.acc_bd_minus_1 <= ConvIntegerBase.float_mantissa:
            pipeline_datatype = torch.float32
        elif self.acc_bd_minus_1 < ConvIntegerBase.double_mantissa:
            pipeline_datatype = torch.float64
        else:
            assert (not 'Accumulator bit depth is too big')
        
        if 'cpu' in str(self.weight.device) and isinstance(self, Conv2di):
            pipeline_datatype = torch.int32

        return pipeline_datatype
                
    def ptflops_custom_hook(self):
        from ptflops.flops_counter import conv_flops_counter_hook
        return conv_flops_counter_hook
    
class Conv2di(ConvIntegerBase, nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.base_conv_function = nn.Conv2d.forward
        nn.Conv2d.__init__(self, *args, **kwargs)
        num_output_channels = args[1]
        ConvIntegerBase.__init__(self, num_output_channels)

    def _get_weights_of_ith_channel(self, i):
        return self.weight.data[i, :, :, :]

    def _get_per_channel_shifts_for_weights(self):
        return self.per_channel_shifts.view(self.out_channels, 1, 1, 1)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if state_dict.get(f'{prefix}is_quantized', False):
            if self.bias is None: #TODO: clean this place after adding bias to all int models
                self.bias = torch.nn.Parameter( torch.zeros( self.out_channels, device = self.weight.device ) )
            
            self.bias.data = self.bias.data.to( dtype=torch.float64 ) #to avoid precision loss in conv2d load state dict
        
        nn.Conv2d._load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        
        if state_dict.get(f'{prefix}is_quantized', False):
            ConvIntegerBase._load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
