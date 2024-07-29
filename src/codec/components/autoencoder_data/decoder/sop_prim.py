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
from typing import Tuple

##
from src.codec.components.activations import ResAU
from src.codec.components.base_layers import (LightResidualBlock, clip_image_sgl,
                                              cropping_layer, conv2x2_pxl2,
                                              denormalize, conv1x1, make_layer, conv3x3)
from ..base import AEBase



class DecoderSOPPrim(AEBase):
    IN_CHS: int = 160
    OUT_CHS: int = 1
    C1: int = 96
    C2: int = 64
    
    def __init__(self, chs_ls=IN_CHS, chs_hidden: Tuple[int] = [C1, C2], *args, **kwargs):    
        super(DecoderSOPPrim, self).__init__()
        assert len(chs_hidden) == 2

        self.first_stage = nn.Sequential(
            make_layer(LightResidualBlock, chs_ls, 1),
            conv2x2_pxl2(chs_ls, chs_hidden[0] - 32, groups=1, bias=False),
            ResAU(chs_hidden[0] - 32, (chs_hidden[0] - 32)//16)
        )
        self.conv2_t = conv2x2_pxl2(chs_hidden[0] - 32, chs_hidden[1] - 32, groups=1, bias=False)
        self.iact2 = ResAU(chs_hidden[1] - 32, (chs_hidden[1] - 32)//16)
        ##
        self.conv3 = conv3x3(chs_hidden[1] - 32, chs_hidden[1] - 32, groups=1, stride=1)
        self.iact3 = ResAU(chs_hidden[1] - 32, (chs_hidden[1] - 32)//16)
        self.conv4 = conv1x1(chs_hidden[1] - 32, self.OUT_CHS * 4 * 4, groups=1, stride=1)
        self.ps = torch.nn.PixelShuffle(4)

        self.out_scale_factor = 16
        self.max_scale_factor = 16

    def forward(self, x, h=0, w=0):
        x = self.first_stage(x)
        x = cropping_layer(x, h, w, depth=3, skip_depth_step=False)

        x = self.conv2_t(x)
        x = cropping_layer(x, h, w, depth=2, skip_depth_step=False)
        x = self.iact2(x)
        x = self.conv3(x)
        x = self.iact3(x)
        x = self.conv4(x)
        x = self.ps(x)

        x = denormalize(x)
        x = clip_image_sgl(x)
        return x
