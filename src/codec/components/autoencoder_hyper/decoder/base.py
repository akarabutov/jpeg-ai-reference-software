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

import math
import torch
import torch.nn as nn

from src.codec.common import determinism_on_eval
from src.codec.components.base_layers import (conv3x3, conv4x4_t, conv1x1, cropping_layer)

from ..base import AEBase

class HyperDecoderBase(AEBase):
    def __init__(self, chs=192, num_out=2, skip_depth_step=False, *args, **kwargs):
        super(HyperDecoderBase, self).__init__()
		# TODO: remove num_out

        self.skip_depth_step = skip_depth_step

        # params = QuantParamHyperDecoder.get_params()
        # parent_name = 'HyperDecoder'
        # Note: to accelerate the inference, this conv1x1i can be fused with the following conv4x4_t in inference.
        self.conv1 = conv1x1(chs, chs, stride=1)

        self.conv2 = conv4x4_t(chs, chs)

        self.conv3 = conv3x3(chs, chs, stride=1)

        self.conv4 = conv3x3(chs, 4 * chs, stride=1)

        self.out_scale_factor = 4
        self.max_scale_factor = 4
        self.act1 = nn.ReLU6()
        self.act2 = nn.ReLU6()

    @determinism_on_eval
    def forward(self, x_in, h=0, w=0):
        x = x_in.to(self.conv1.weight.dtype)
        
        x = self.conv1(x)

        x = self.conv2(x) # deconv
        x = cropping_layer(x, h, w, depth=5, skip_depth_step=self.skip_depth_step)
        x = self.act1(x)

        x = self.conv3(x)
        x = self.act2(x)

        x = self.conv4(x) # conv
        #if self.skip_depth_step:
        #    x = ContextUtils.up_shuffle(torch.chunk(x, 4, dim=1))
        #    x = cropping_layer(x, h, w, depth=4, skip_depth_step=self.skip_depth_step)

        if x.dtype == torch.float64:
            x = x.to(dtype=torch.float32)
        return x
