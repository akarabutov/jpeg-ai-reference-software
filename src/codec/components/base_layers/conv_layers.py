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

from torch.nn import Conv2d, ConvTranspose2d
from .conv_quant_layers import Conv2di
from .utils import initialize_weights

def conv1x1(in_chs, out_chs, stride=1, groups=1, bias=False):
    return  Conv2d(in_chs,
                   out_chs,
                   kernel_size=1,
                   stride=stride,
                   groups=groups,
                   padding=0,
                   bias=bias)

def conv1x1i(in_chs, out_chs, stride=1, groups=1, bias=False):
    return Conv2di(in_chs,
                   out_chs,
                   kernel_size=1,
                   stride=stride,
                   groups=groups,
                   padding=0,
                   bias=bias)

def conv3x3(in_chs, out_chs, stride=2, groups=1, bias=True):
    return  Conv2d(in_chs,
                   out_chs,
                   kernel_size=3,
                   stride=stride,
                   groups=groups,
                   padding=1,
                   bias=bias)

def conv3x3i(in_chs, out_chs, stride=2, groups=1, bias=True):
    return Conv2di(in_chs,
                   out_chs,
                   kernel_size=3,
                   stride=stride,
                   groups=groups,
                   padding=1,
                   bias=bias)

def conv3x3_t(in_chs,
              out_chs,
              stride=2,
              bias=True):
    return ConvTranspose2d( in_chs,
                            out_chs,
                            kernel_size=3,
                            stride=stride,
                            padding=1,
                            output_padding=1,
                            bias=bias)

def conv4x4_t(in_chs,
              out_chs,
              stride=2,
              groups: int = 1,
              bias=True):
    return ConvTranspose2d( in_chs,
                            out_chs,
                            kernel_size=4,
                            stride=stride,
                            groups=groups,
                            padding=1,
                            bias=bias)

class conv2x2_pxl2(nn.Module):
    def __init__(self, in_chs, out_chs, stride=1, groups=1, bias=True):
        super(conv2x2_pxl2, self).__init__()
        self.conv = Conv2d (in_chs,
                            out_chs * 4,
                            kernel_size=2,
                            stride=stride,
                            groups=groups,
                            padding=0,
                            bias=bias)
        self.pxl = nn.PixelShuffle(2)

    def forward(self, x):
        x = nn.functional.pad(x, (0, 1, 0, 1), "constant", 0)
        x = self.conv(x)
        x = self.pxl(x)
        return x

class ResidualBlock_BN(nn.Module):
    '''Residual block with BN and scale
    ---Conv-BN-ReLU-Conv-BN-+-
         |__________________|
    '''
    def __init__(self, nf=64):
        super(ResidualBlock_BN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.BN1 = nn.BatchNorm2d(nf)
        self.BN2 = nn.BatchNorm2d(nf)
        self.scale = nn.Parameter(torch.FloatTensor([1e-1]))
        # initialization
        initialize_weights([self.conv1, self.conv2, self.BN1, self.BN2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.BN1(self.conv1(x)), inplace=True)
        out = self.BN2(self.conv2(out))
        return identity + out * self.scale


class ResidualBlock_BN_RectKernel(nn.Module):
    '''Residual block with BN and scale
    ---Conv-BN-ReLU-Conv-BN-+-
         |__________________|
    '''
    def __init__(self, nf=64):
        super(ResidualBlock_BN_RectKernel, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, [1, 3], 1, [0, 1], bias=True)
        self.conv2 = nn.Conv2d(nf, nf, [3, 1], 1, [1, 0], bias=True)
        self.BN1 = nn.BatchNorm2d(nf)
        self.BN2 = nn.BatchNorm2d(nf)
        self.scale = nn.Parameter(torch.FloatTensor([1e-1]))
        # initialization
        initialize_weights([self.conv1, self.conv2, self.BN1, self.BN2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.BN1(self.conv1(x)), inplace=True)
        out = self.BN2(self.conv2(out))
        return identity + out * self.scale

class LightResidualBlock(nn.Module):
    '''Residual block with one_Conv
    ---Conv-ReLU-+-
         |_______|
    '''
    def __init__(self, planes):
        super(LightResidualBlock, self).__init__()
        self.conv1 = conv3x3(planes, planes, stride=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        return out + x

class LightCombineBlock(nn.Module):
    '''LightCombineBlock
        input: cat(sup_latent, latent, dim=1)
        output: latent
    '''
    def __init__(self, chs_ls, chs_supp):
        super(LightCombineBlock, self).__init__()
        self.chs_ls = chs_ls
        self.conv1 = conv3x3(chs_ls + chs_supp, int(chs_ls/2), stride=1, groups=1)

    def forward(self, x):
        info = self.conv1(x)
        x = torch.cat((info, x[:, -self.chs_ls:]), dim=1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, planes):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(planes, planes, stride=1)
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return out + x

class MaskedConv2d(Conv2d):
    """Masked Conv2d
    """
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.register_mask(mask_type)

    def forward(self, x):
        _, _, H, W = self.weight.size()
        for i in range(H // 2):
            self.mask[:, :, i, W - 1 - i:] = 0
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

    def register_mask(self, mask_type):
        """register the mask for convolution
        ------------------------------------
        |  1       1       1       1       0 |
        |  1       1       1       0       0 |
        |  1       1    1 if B     0       0 |   h // 2
        |  0       0       0       0       0 |   h // 2 + 1
        |  0       0       0       0       0 |   end - 1
         ------------------------------------
           0       1     w//2    w//2+1    end - 1

        Args:
            mask_type:

        Returns:
            none:
        """
        o = self.check_mask_type(mask_type)
        _, _, h, w = self.weight.size()

        self.register_buffer('mask', self.weight.data.clone())
        self.mask.fill_(1)
        self.mask[:, :, (h // 2 + 0):, (w // 2 + o):] = 0
        self.mask[:, :, (h // 2 + 1):, :] = 0
        for i in range(h // 2):
            self.mask[:, :, i, w - 1 - i:] = 0

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys,
                              unexpected_keys, error_msgs):
        super(MaskedConv2d, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                        missing_keys, unexpected_keys, error_msgs)

    @staticmethod
    def check_mask_type(mask_type):
        mask_dict = {
            'A': 0,
            'B': 1,
        }

        if mask_type not in mask_dict:
            raise AssertionError('Invalid mask_type={}'.format(mask_type))

        offset = mask_dict[mask_type]
        return offset
