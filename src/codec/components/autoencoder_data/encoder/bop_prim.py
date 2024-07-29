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
from typing import Tuple

##
from src.codec.components.activations import ResAU
from src.codec.components.base_layers import (conv1x1, conv3x3, normalize, padding_layer, feature_clipping)

from ..base import AEBase


class EncoderBOPPrim(AEBase):
    CHS_in: int = 1
    CHS_LS: int = 160
    C1: int = 128
    
    def __init__(self, chs_in: int = CHS_in, chs_ls: int =CHS_LS, chs_hidden: Tuple[int] = [C1], *args, **kwargs):    
        super(EncoderBOPPrim, self).__init__()
        self.conv1 = conv3x3(chs_in, chs_hidden[0])
        self.act1 = ResAU(chs_hidden[0])

        self.conv2 = conv3x3(chs_hidden[0], chs_hidden[0])
        self.act2 = ResAU(chs_hidden[0])
        ##
        self.conv3 = conv3x3(chs_hidden[0], chs_hidden[0])
        self.act3 = ResAU(chs_hidden[0])
        ##
        self.conv4 = conv3x3(chs_hidden[0], chs_ls)
        self.conv5 = conv1x1(chs_ls, chs_ls)
        ##
        self.out_scale_factor = 16
        self.max_scale_factor = 16

        self.clip_thres = None

    def forward(self, x, alfa=1, h=0, w=0, is_collect=False, is_clip=False):
        x = normalize(x)
        x = padding_layer(x, h, w, depth=0, skip_depth_step=False)

        x = self.conv1(x)
        x1 = x
        x = feature_clipping(x, 'E1B', self.clip_thres, is_clip)
        x = self.act1(x)
        x = padding_layer(x, h, w, depth=1, skip_depth_step=False)

        x = self.conv2(x)
        x2 = x
        x = feature_clipping(x, 'E2B', self.clip_thres, is_clip)
        x = self.act2(x)
        x = padding_layer(x, h, w, depth=2, skip_depth_step=False)

        x = self.conv3(x)
        x3 = x
        x = feature_clipping(x, 'E3B', self.clip_thres, is_clip)
        x = self.act3(x)
        x = padding_layer(x, h, w, depth=3, skip_depth_step=False)
        x = self.conv4(x)
        x4 = x
        x = feature_clipping(x, 'E4B', self.clip_thres, is_clip)
        x = self.conv5(x)
        x = feature_clipping(x, 'E5B', self.clip_thres, is_clip)
        if is_collect:
            return x1, x2, x3, x4, x
        else:
            return x
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict_module = super(EncoderBOPPrim, self).state_dict(destination, prefix, keep_vars)

        if self.clip_thres is not None:
            state_dict_thres = {prefix+'clip_thres': self.clip_thres}
            state_dict_module.update(state_dict_thres)

        return state_dict_module

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super(EncoderBOPPrim, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        
        if state_dict.get(f'{prefix}clip_thres', False):
            self.clip_thres = state_dict[prefix+'clip_thres']

        if prefix+'clip_thres' in unexpected_keys:
            unexpected_keys.remove(prefix+'clip_thres')
