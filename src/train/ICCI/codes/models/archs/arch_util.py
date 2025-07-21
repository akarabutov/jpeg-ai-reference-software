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
import torch.nn.init as init
import torch.nn.functional as F


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        #initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class Wide_activation(nn.Module):
    def __init__(self, K = 72, M = 3*72):
        super(Wide_activation, self).__init__()
        self.conv1 = nn.Conv2d(K, M, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(M, K, 1, 1, 0, bias=True)
        self.conv3 = nn.Conv2d(K, K, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)

    def forward(self, x):
        out = self.lrelu(self.conv1(x))
        out = self.conv2(out)
        return self.conv3(out)+x

class ResidualBlock_BN(nn.Module):
    '''Residual block with BN and scale
    ---Conv-BN-ReLU-Conv-BN-+-
         |__________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_BN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.BN1   = nn.BatchNorm2d(nf)
        self.BN2   = nn.BatchNorm2d(nf)
        self.scale = nn.Parameter(torch.FloatTensor([1e-1]))
        # initialization
        initialize_weights([self.conv1, self.conv2, self.BN1, self.BN2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.BN1(self.conv1(x)), inplace=True)
        out = self.BN2(self.conv2(out))
        return identity + out*self.scale

class ResidualBlock_BN_RectKernel(nn.Module):
    '''Residual block with BN and scale
    ---Conv-BN-ReLU-Conv-BN-+-
         |__________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_BN_RectKernel, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, [1,3], 1, [0,1], bias=True)
        self.conv2 = nn.Conv2d(nf, nf, [3,1], 1, [1,0], bias=True)
        self.BN1   = nn.BatchNorm2d(nf)
        self.BN2   = nn.BatchNorm2d(nf)
        self.scale = nn.Parameter(torch.FloatTensor([1e-1]))
        # initialization
        initialize_weights([self.conv1, self.conv2, self.BN1, self.BN2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.BN1(self.conv1(x)), inplace=True)
        out = self.BN2(self.conv2(out))
        return identity + out*self.scale

class ResidualBlock_BN_RectKernel_1551(nn.Module):
    '''Residual block with BN and scale
    ---Conv-BN-ReLU-Conv-BN-+-
         |__________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_BN_RectKernel_1551, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, [1,5], 1, [0,2], bias=True)
        self.conv2 = nn.Conv2d(nf, nf, [5,1], 1, [2,0], bias=True)
        self.BN1   = nn.BatchNorm2d(nf)
        self.BN2   = nn.BatchNorm2d(nf)
        self.scale = nn.Parameter(torch.FloatTensor([1e-1]))
        # initialization
        initialize_weights([self.conv1, self.conv2, self.BN1, self.BN2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.BN1(self.conv1(x)), inplace=True)
        out = self.BN2(self.conv2(out))
        return identity + out*self.scale

class CONV_BN_RELU(nn.Module):

    def __init__(self, nf=64):
        super(CONV_BN_RELU, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.BN   = nn.BatchNorm2d(nf)
        # initialization
        initialize_weights([self.conv, self.BN], 0.1)

    def forward(self, x):
        out = self.BN(self.conv(x))
        out = F.relu(out, inplace=True)
        return out


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output



class ConvNextBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 2 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = GELU()
        self.pwconv2 = nn.Linear(2 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNextStem(nn.Module):

    def __init__(self, dim_i, dim_o):
        super().__init__()
        self.conv = nn.Conv2d(dim_i, dim_o, 3, 1, 1, bias=True)
        self.norm = nn.LayerNorm(dim_o, eps=1e-6)
        self.act = GELU()

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return x
    

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

import math
class GELU_(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_