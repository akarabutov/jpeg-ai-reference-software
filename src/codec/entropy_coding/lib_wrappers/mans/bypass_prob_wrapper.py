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

from .base_prob_wrapper import BaseProbWrapper
 
import numpy as np
import torch


class BypassProbWrapper(BaseProbWrapper):
    def __init__(self, backend):
        super(BypassProbWrapper, self).__init__(backend)
        
    @staticmethod
    def calc_precision(max_symbol_value:int) -> int:
        return int(np.ceil(np.log2(max_symbol_value + 1)))
    
    def get_pmf(self, max_symbol_value=1):
        symbol_num = max_symbol_value+1
        norm_pmf = (torch.arange(symbol_num)+1) / (0.5 * (max_symbol_value+1) * max_symbol_value) 
        return norm_pmf
        
 
    def encode(self, x, max_symbol_value=1, name=None):
        """
 
        Args:
            x: shape=[C, H, W], [B=1, C, H, W], [B=1*C*H*W], dtype=int
            max_symbol_value:
            name:
 
        Returns:
            none:
        """
        x = self.convert_data(x).cpu().numpy().ravel()
        pmf = self.get_pmf(max_symbol_value)
        self.backend.encode_factorize(pmf.cpu().numpy(), x)
 
    def decode(self, shape, max_symbol_value=1, device=torch.device('cpu'), name=None):
        """
 
        Args:
            shape: value=[C, H, W], [B=1, C, H, W], [B=1*C*H*W], dtype=int
            max_symbol_value:
            device:
            name:
 
        Returns:
            none:
        """
        x = np.zeros(shape, dtype=np.uint32).ravel()
        C = shape[-3]
        pmf = self.get_pmf(max_symbol_value)
        #pmf = pmf.view(1,1,-1).repeat(C, 1, 1)
        self.backend.decode_factorize(pmf, x)
        x = torch.tensor(x.astype(np.int32), device = device)
        x = x.view(shape)
        return x
