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
import numbers
from .conv_layers import conv3x3, conv3x3_t, conv1x1
from .base_funcs import Reshape, Norm, MatrixMult
from .utils import make_layer

__all__ = ['TAM']

##########################################################################
## Layer Norm

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

    



##########################################################################
class FeedForward(nn.Module):
    def __init__(self, dim, gamma, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*gamma)

        
        self.reshape1 = Reshape("b c h w", "b (h w) c")
        self.norm1 = LayerNorm(dim)
        self.reshape2 = Reshape("b (h w) c", "b c h w")

        self.project_in = conv1x1(dim, hidden_features*2, bias=bias)
        self.dwconv = conv3x3(hidden_features*2, hidden_features*2, stride=1, groups=hidden_features*2, bias=bias)
        self.act1 = nn.ELU()
        self.project_out = conv1x1(hidden_features, dim, bias=bias)

    def forward(self, x):
        h,w = x.shape[-2:]
        x = self.reshape1(x)
        x = self.norm1(x)
        x = self.reshape2(x, h=h, w=w)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act1(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
class Attention(nn.Module):
    def __init__(self, dim, n=4, bias=False):
        super(Attention, self).__init__()
        self.num_heads = n
        self.temperature = nn.Parameter(torch.ones(n, 1, 1))
        self.project_out = conv1x1(dim, dim, bias=bias)     
        self.reshape1 = Reshape('b (head c) h w', f'b head c (h w)')
        self.reshape2 = Reshape('b head c (h w)',  f'b (head c) h w')
        self.norm1 = Norm(-1)
        self.mm = MatrixMult()
        self.softmax1 = nn.Softmax(dim=-1)
    
    
    def forward(self, x):
        h,w=x.shape[-2:]
        branch1, branch2, branch3 = x.chunk(3, dim=1)   
        
        branch1 = self.reshape1(branch1, head=self.num_heads)
        branch2 = self.reshape1(branch2, head=self.num_heads)
        branch3 = self.reshape1(branch3, head=self.num_heads)

        branch1 = self.norm1(branch1)
        branch2 = self.norm1(branch2).transpose(-2, -1)

        attn = self.mm(branch1, branch2)
        attn *= self.temperature
        attn = self.softmax1(attn)
        x = self.mm(attn, branch3)    

        x = self.reshape2(x, h=h, w=w, head=self.num_heads)
        x = self.project_out(x)
        return x

class PrepareData(nn.Module):
    def __init__(self, dim, bias=False):
        super(PrepareData, self).__init__()
        self.reshape1 = Reshape("b c h w", "b (h w) c")
        self.norm1 = LayerNorm(dim)
        self.reshape2 = Reshape("b (h w) c", "b c h w")
        self.conv1 =  conv1x1(dim, dim*3, stride=1, bias=bias)
        self.conv2 = conv3x3(dim*3, dim*3, stride=1, groups=dim*3, bias=bias)

    def forward(self, x):
        h,w = x.shape[-2:]
        x = self.reshape1(x)
        x = self.norm1(x)
        x = self.reshape2(x,h=h,w=w)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
        

class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super(TransformerBlock, self).__init__()
        
        n = 4 
        gamma = 4
        bias = False        

        self.prep_data = PrepareData(dim, bias)
        self.attn = Attention(dim, n, bias)
        self.ffn = FeedForward(dim, gamma, bias)

    def forward(self, x):
        tmp = self.prep_data(x)
        x = x + self.attn(tmp)
        x = x + self.ffn(x)

        return x

class TAM(nn.Module):
    def __init__(self, dim,  ds_atten_module: bool = False):
        super(TAM, self).__init__()
        num_blks = 2 
        
        self.pre_downsample = ds_atten_module
        self.post_upsample = ds_atten_module
        if self.pre_downsample:
            self.ds_conv = conv3x3(dim, dim, stride=2)
        if self.post_upsample:
            self.us_conv = conv3x3_t(dim, dim,stride=2)
            
        self.TABs = make_layer(TransformerBlock, dim, num_blks)

    def forward(self, x):
        if self.pre_downsample:
            x = self.ds_conv(x)
        x = self.TABs(x)
        if self.post_upsample:
            x = self.us_conv(x)
        return x

