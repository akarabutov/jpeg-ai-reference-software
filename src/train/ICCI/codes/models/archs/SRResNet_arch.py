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

import functools

import models.archs.arch_util as arch_util
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSRResNet(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(MSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.recon_trunk = arch_util.make_layer(basic_block, nb)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.initialize_weights([self.conv_first, self.HRconv, self.conv_last], 0.1)
        if self.upscale > 1:
            arch_util.initialize_weights(self.upconv1, 0.1)
        if self.upscale == 4:
            arch_util.initialize_weights(self.upconv2, 0.1)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out = self.recon_trunk(fea)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out

   
class ThreeStageYUVLite_DWT2_444_jointUV(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nf=48, nbY=2, nbUV=2):
        super(ThreeStageYUVLite_DWT2_444_jointUV, self).__init__()
        init_block_list = []
        basic_block = functools.partial(arch_util.ResidualBlock_BN_RectKernel, nf=nf)

        self.conv_first_Y = nn.Conv2d(48, nf, 3, 1, 1, bias=True)
        init_block_list.append(self.conv_first_Y)
        self.BNY = nn.BatchNorm2d(nf)
        init_block_list.append(self.BNY)
        self.hidden_layer_Y = arch_util.make_layer(basic_block, nbY)

        self.conv_last_Y = nn.Conv2d(nf, 16, 3, 1, 1, bias=True)
        init_block_list.append(self.conv_last_Y)
        self.scaleY = nn.Parameter(torch.FloatTensor([1e-1]))


        self.conv_firstUV = nn.Conv2d(48, nf, 3, 1, 1, bias=True)
        init_block_list.append(self.conv_firstUV)
        self.BNUV = nn.BatchNorm2d(nf)
        init_block_list.append(self.BNUV)
        self.hidden_layer_UV = arch_util.make_layer(basic_block, nbUV)

        self.conv_last_u = nn.Conv2d(nf, 16, 3, 1, 1, bias=True)
        self.conv_last_v = nn.Conv2d(nf, 16, 3, 1, 1, bias=True)
        init_block_list.append(self.conv_last_u)
        init_block_list.append(self.conv_last_v)

        self.scaleU = nn.Parameter(torch.FloatTensor([1e-1]))
        self.scaleV = nn.Parameter(torch.FloatTensor([1e-1]))

        arch_util.initialize_weights(init_block_list, 0.1)

    def forward(self, x):
        Y = x[:,0:1,:,:]
        U = x[:,1:2,:,:]
        V = x[:,2:3,:,:]
        
        dwt2_res = self.process_dwt_2(Y, U, V)

        # Network1
        final_Y = self.process_Y(*dwt2_res)

        # Network2
        final_UV= self.process_UV(*dwt2_res)

        # Network2
        # final_V= self.process_V(*dwt2_res)

        return torch.cat((final_Y, final_UV), axis=1)

    def process_Y(self, Y16, U16, V16, YUV16):
        # Network1
        fea_Y = F.relu(self.BNY(self.conv_first_Y(YUV16)), inplace=True)
        fea_Y = self.hidden_layer_Y(fea_Y)
        out_Yres = self.conv_last_Y(fea_Y)
        final_Y16 = out_Yres * self.scaleY + Y16
        final_Y = self.my_tf_idwt_2(final_Y16)

        return final_Y

    def process_U(self, Y16, U16, V16, YUV16):
        # Network2
        #UV
        fea_uv = F.relu(self.BNUV(self.conv_firstUV(YUV16)), inplace=True)
        fea_uv = self.hidden_layerUV(fea_uv)

        #network U
        out_Ures = self.conv_last_u(fea_uv)
        final_U16 = out_Ures * self.scaleU + U16

        #IDWT
        final_U = self.my_tf_idwt_2(final_U16)

        return final_U

    def process_V(self, Y16, U16, V16, YUV16):
        # Network3
        #V
        fea_uv = F.relu(self.BNUV(self.conv_firstUV(YUV16)), inplace=True)
        fea_uv = self.hidden_layerUV(fea_uv)

        #network V
        out_Vres = self.conv_last_v(fea_uv)
        final_V16 = out_Vres * self.scaleV + V16

        #IDWT
        final_V = self.my_tf_idwt_2(final_V16)

        return final_V
    
    def process_UV(self, Y16, U16, V16, YUV16):
        # Network2
        #UV
        fea_uv = F.relu(self.BNUV(self.conv_firstUV(YUV16)), inplace=True)
        fea_uv = self.hidden_layer_UV(fea_uv)

        #network U
        out_Ures = self.conv_last_u(fea_uv)
        final_U16 = out_Ures * self.scaleU + U16
        #network V
        out_Vres = self.conv_last_v(fea_uv)
        final_V16 = out_Vres * self.scaleV + V16
        
        #IDWT
        final_U = self.my_tf_idwt_2(final_U16)
        final_V = self.my_tf_idwt_2(final_V16)
        final_UV = torch.cat((final_U, final_V), axis=1)

        return final_UV
    
    def my_tf_dwt(self, x):
        x01 = x[:,:,0::2,:] / 2.0
        x02 = x[:,:,1::2,:] / 2.0
        x1 = x01[:,:,:,0::2]
        x2 = x02[:,:,:,0::2]
        x3 = x01[:,:,:,1::2]
        x4 = x02[:,:,:,1::2]
        x_LL =  x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH =  x1 - x2 - x3 + x4
        return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

    def my_tf_dwt_2(self, x4):
        x16_list = []
        for i in range(4):
            cur_x_dwt2 = self.my_tf_dwt(x4[:, i:i+1, :, :])
            x16_list.append(cur_x_dwt2)

        x16 = torch.cat(x16_list, dim=1)
        return x16

    def my_tf_idwt(self, x):
        n, c, h, w = x.size()
        out_channel = c//4
        x1 = x[:, 0:out_channel, :, :] / 2.0
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2.0
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2.0
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2.0
        rec1 = x1 - x2 - x3 + x4
        rec2 = x1 - x2 + x3 - x4
        rec3 = x1 + x2 - x3 - x4
        rec4 = x1 + x2 + x3 + x4
        h = torch.ones(n, out_channel, h*2, w*2).cuda()
        h[:,:,0::2, 0::2] = rec1
        h[:,:,1::2, 0::2] = rec2
        h[:,:,0::2, 1::2] = rec3
        h[:,:,1::2, 1::2] = rec4
        return h

    def my_tf_idwt_2(self, x16):

        x4_list = []
        for i in range(4):
            cur_x_idwt = self.my_tf_idwt(x16[:, i * 4: (i + 1) * 4, :, :])
            x4_list.append(cur_x_idwt)

        x4 = torch.cat(x4_list, dim=1)
        x = self.my_tf_idwt(x4)

        return x

    def process_dwt(self, Y, U, V):
        Y444 = self.my_tf_dwt(Y)
        U4 = self.my_tf_dwt(U)
        V4 = self.my_tf_dwt(V)
        YUV = torch.cat((Y444, U4, V4), 1)
        return Y444, U4, V4, YUV

    def process_dwt_2(self, Y, U, V):

        Y16_list = []
        U16_list = []
        V16_list = []

        Y4, U4, V4, _ = self.process_dwt(Y, U, V)
        for i in range(4):
            cur_y_dwt2 = self.my_tf_dwt(Y4[:, i:i+1, :, :])
            Y16_list.append(cur_y_dwt2)

            cur_u_dwt2 = self.my_tf_dwt(U4[:, i:i+1, :, :])
            U16_list.append(cur_u_dwt2)

            cur_v_dwt2 = self.my_tf_dwt(V4[:, i:i + 1, :, :])
            V16_list.append(cur_v_dwt2)

        Y16 = torch.cat(Y16_list, dim=1)
        U16 = torch.cat(U16_list, dim=1)
        V16 = torch.cat(V16_list, dim=1)

        YUV16 = torch.cat([Y16, U16, V16], dim=1)

        return Y16, U16, V16, YUV16

    def process_dwt444(self, yuv):
        Y444 = self.my_tf_dwt(yuv[:, 0:1])
        U4 = self.my_tf_dwt(yuv[:, 1:2])
        V4 = self.my_tf_dwt(yuv[:, 2:3])
        YUV = torch.cat((Y444, U4, V4), 1)
        return Y444, U4, V4, YUV

    def process_dwt444_2(self, yuv):
        Y4   = self.my_tf_dwt(yuv[:, 0:1])
        U4   = self.my_tf_dwt(yuv[:, 1:2])
        V4   = self.my_tf_dwt(yuv[:, 2:3])

        Y16 = self.my_tf_dwt_2(Y4)
        U16 = self.my_tf_dwt_2(U4)
        V16 = self.my_tf_dwt_2(V4)

        YUV16 = torch.cat((Y16, U16, V16), 1)

        return Y16, U16, V16, YUV16

    def process_comp(self, Y16, U16, V16, YUV16, comp_idx: int) -> torch.Tensor:
        cmd = [self.process_Y, self.process_UV, self.process_UV]
        assert comp_idx >= 0 and comp_idx < len(cmd)
        return cmd[comp_idx](Y16, U16, V16, YUV16)
