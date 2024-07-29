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

from .conv_layers import ResidualBlock, conv3x3, conv3x3_t
from .utils import make_layer

try:
    from torch.cuda.amp import autocast
    HAS_AMP = True
except:  # noqa: E722
    HAS_AMP = False


class CAB(nn.Module):
    def __init__(self, planes):
        super(CAB, self).__init__()
        self.residual_trunk = make_layer(ResidualBlock, planes, 2)
        self.residual_mask2 = make_layer(ResidualBlock, planes, 1)
        self.residual_mask3 = make_layer(ResidualBlock, planes, 1)
        self.subscale = conv3x3(planes, planes, stride=2)
        self.upscale = conv3x3_t(planes, planes)


    def forward(self, x, gama):
        trunk_branch = self.residual_trunk(x)
        ##
        mask_branch = self.subscale(x)
        mask_branch = self.residual_mask2(mask_branch)
        if HAS_AMP and self.training:
            with autocast(enabled=False):
                mask_branch = self.residual_mask3(mask_branch.to(dtype=torch.float32))
                mask_branch = self.upscale(mask_branch)
        else:
            mask_branch = self.residual_mask3(mask_branch)
            mask_branch = self.upscale(mask_branch)
        mask_branch = mask_branch.sigmoid()
        ##
        trunk_branch = gama * trunk_branch.mul(mask_branch)

        y = trunk_branch + x
        return y