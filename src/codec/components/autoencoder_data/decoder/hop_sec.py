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
from src.codec.components.base_layers import (clip_image_sgl,
                                              conv3x3_t, cropping_layer, denormalize,
                                              conv1x1, LightCombineBlock,
                                              CAB, TAM)
from ..base import AEBase

class DecoderHOPSec(AEBase):
    IN_CHS: int = 48
    IN_SUPP_CHS: int = 160
    OUT_CHS: int = 2
    C1: int = 64
    C2: int = 64
    
    def __init__(self, chs_ls=IN_CHS, chs_supp: int = IN_SUPP_CHS, chs_hidden: Tuple[int] = [C1, C2], *args, **kwargs):    
        super(DecoderHOPSec, self).__init__()
        assert len(chs_hidden) == 2
            
        self.first_stage = LightCombineBlock(chs_ls, chs_supp)
        self.conv2_t = conv3x3_t(int(chs_ls*1.5), chs_hidden[0])
        
        ## Attention modules
        self.CAB = CAB(chs_hidden[0])
        self.TAM = TAM(dim=chs_hidden[1], ds_atten_module=False)
                
        self.iact2 = ResAU(chs_hidden[0])
        ##
        self.conv3_t = conv1x1(chs_hidden[0], chs_hidden[1]*4, bias=True)
        self.pix_sfl3 = torch.nn.PixelShuffle(2)
        self.iact3 = ResAU(chs_hidden[1])
        ##
        ##
        self.conv4_t = conv3x3_t(chs_hidden[1], self.OUT_CHS*4)
        self.pix_sfl4 = torch.nn.PixelShuffle(2)
        self.out_scale_factor = 16
        self.max_scale_factor = 16

    def forward(self, x, alfa=1, h=0, w=0):

        x = self.first_stage(x)

        x = self.conv2_t(x)
        x = self.CAB(x, alfa)
        x = cropping_layer(x, h, w, depth=3, skip_depth_step=True)
        x = self.iact2(x)

        x = self.conv3_t(x)
        x = self.pix_sfl3(x)
        x = self.TAM(x)
        x = cropping_layer(x, h, w, depth=2, skip_depth_step=True)
        x = self.iact3(x)

        x = self.conv4_t(x)
        x = self.pix_sfl4(x)

        x = denormalize(x)
        x = clip_image_sgl(x)

        return x
