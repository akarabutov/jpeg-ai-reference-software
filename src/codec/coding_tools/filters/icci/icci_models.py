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
import torch.nn.functional as F

##
from src.codec.components.base_layers import (ResidualBlock_BN_RectKernel, make_layer)
from src.codec.components.base_layers.utils import initialize_weights


class ThreeStageYUVLite_DWT_V3_444(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nf=48, nbY=8, nbUV=13):
        super(ThreeStageYUVLite_DWT_V3_444, self).__init__()

        self.conv_first_Y = nn.Conv2d(in_nc * 12, nf, 3, 1, 1, bias=True)
        self.BNY = nn.BatchNorm2d(nf)
        self.hidden_layer_Y = make_layer(ResidualBlock_BN_RectKernel, nf, nbY)
        self.conv_last_Y = nn.Conv2d(nf, 4, 3, 1, 1, bias=True)
        self.scaleY = nn.Parameter(torch.FloatTensor([1e-1]))

        self.conv_first_U = nn.Conv2d(12, nf, 3, 1, 1, bias=True)
        self.BNU = nn.BatchNorm2d(nf)
        self.hidden_layer_U = make_layer(ResidualBlock_BN_RectKernel, nf, nbUV)
        self.conv_last_U = nn.Conv2d(nf, 4, 3, 1, 1, bias=True)
        self.scaleU = nn.Parameter(torch.FloatTensor([1e-1]))

        self.conv_first_V = nn.Conv2d(12, nf, 3, 1, 1, bias=True)
        self.BNV = nn.BatchNorm2d(nf)
        self.hidden_layer_V = make_layer(ResidualBlock_BN_RectKernel, nf, nbUV)
        self.conv_last_V = nn.Conv2d(nf, 4, 3, 1, 1, bias=True)
        self.scaleV = nn.Parameter(torch.FloatTensor([1e-1]))

        initialize_weights(
            [self.conv_first_Y, self.conv_last_Y, self.conv_first_U, self.conv_last_U], 0.1)
        initialize_weights([self.conv_first_V, self.conv_last_V, self.BNY, self.BNU, self.BNV],
                           0.1)

    def forward(self, Y, U, V):

        dwt_res = self.process_dwt(Y, U, V)
        # Network1
        final_Y = self.process_Y(*dwt_res)
        # Network2
        final_U = self.process_U(*dwt_res)
        # Network3
        final_V = self.process_V(*dwt_res)
        return final_Y, final_U, final_V

    def process_comp(self, Y444, U4, V4, YUV, comp_idx: int) -> torch.Tensor:
        cmd = [self.process_Y, self.process_U, self.process_V]
        assert comp_idx >= 0 and comp_idx < len(cmd)
        return cmd[comp_idx](Y444, U4, V4, YUV)

    def process_dwt444(self, yuv):
        Y444 = self.my_tf_dwt(yuv[:, 0:1])
        U4 = self.my_tf_dwt(yuv[:, 1:2])
        V4 = self.my_tf_dwt(yuv[:, 2:3])
        YUV = torch.cat((Y444, U4, V4), 1)
        return Y444, U4, V4, YUV

    def process_dwt(self, Y, U, V):
        Y444 = self.my_tf_dwt(Y)
        U4 = self.my_tf_dwt(U)
        V4 = self.my_tf_dwt(V)
        YUV = torch.cat((Y444, U4, V4), 1)
        return Y444, U4, V4, YUV

    def process_Y(self, Y444, U4, V4, YUV):
        # Network1
        fea_Y = F.relu(self.BNY(self.conv_first_Y(YUV)), inplace=True)
        fea_Y = self.hidden_layer_Y(fea_Y)
        out_Yres = self.conv_last_Y(fea_Y)
        final_Y = self.my_tf_idwt(out_Yres * self.scaleY + Y444)
        return final_Y

    def process_U(self, Y444, U4, V4, YUV):
        # Network2
        fea_U = F.relu(self.BNU(self.conv_first_U(YUV)), inplace=True)
        fea_U = self.hidden_layer_U(fea_U)
        out_Ures = self.conv_last_U(fea_U)
        final_U = self.my_tf_idwt(out_Ures * self.scaleU + U4)
        return final_U

    def process_V(self, Y444, U4, V4, YUV):
        # Network3
        fea_V = F.relu(self.BNV(self.conv_first_V(YUV)), inplace=True)
        fea_V = self.hidden_layer_V(fea_V)
        out_Vres = self.conv_last_V(fea_V)
        final_V = self.my_tf_idwt(out_Vres * self.scaleV + V4)
        return final_V

    def my_tf_dwt(self, x):
        x01 = x[:, :, 0::2, :] / 2.0
        x02 = x[:, :, 1::2, :] / 2.0
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

    def my_tf_idwt(self, x):
        n, c, h, w = x.size()
        out_channel = c // 4
        x1 = x[:, 0:out_channel, :, :] / 2.0
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2.0
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2.0
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2.0
        rec1 = x1 - x2 - x3 + x4
        rec2 = x1 - x2 + x3 - x4
        rec3 = x1 + x2 - x3 - x4
        rec4 = x1 + x2 + x3 + x4
        h = torch.ones(n, out_channel, h * 2, w * 2, device=x.device)
        h[:, :, 0::2, 0::2] = rec1
        h[:, :, 1::2, 0::2] = rec2
        h[:, :, 0::2, 1::2] = rec3
        h[:, :, 1::2, 1::2] = rec4
        return h
