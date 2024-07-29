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
from einops import rearrange

class Norm(nn.Module):
    def __init__(self, dim: int):
        super(Norm, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        return F.normalize(x, dim=self.dim)
    
    @staticmethod
    def ptflops_calc(module, input, output) -> None:
        module.__flops__ += int(input[0].numel())
        
    
    def ptflops_custom_hook(self):
        return Norm.ptflops_calc    
    
class Reshape(nn.Module):
    def __init__(self, init_dim: str, out_dim: str):
        super(Reshape, self).__init__()
        self.init_dim = init_dim
        self.out_dim = out_dim
        
    def forward(self, x, *args, **kwargs):
        return rearrange(x, f'{self.init_dim} -> {self.out_dim}', *args, **kwargs)
    
class MatrixMult(nn.Module):
    def forward(self, x1, x2):
        return x1 @ x2
        
    @staticmethod
    def ptflops_calc(module, input, output) -> None:
        i1 = input[0]
        i2 = input[1]
        module.__flops__ += int(i1.shape[-2] * i1.shape[-1] * i2.shape[-1] * output.shape[-3] * output.shape[-4])
            
    def ptflops_custom_hook(self):
        return MatrixMult.ptflops_calc    