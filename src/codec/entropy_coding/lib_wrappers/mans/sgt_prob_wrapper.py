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

import numpy as np
import torch
 
from src.codec.components.entropy_coding import GMProbModel
 
from .base_prob_wrapper import BaseProbWrapper
import hashlib
from src.codec.common import TensorOps
 
class SgtProbWrapper(BaseProbWrapper):
    def __init__(self, backend):
        super(SgtProbWrapper, self).__init__(backend)
 
    @staticmethod
    def get_model(entropy_model=None):
        return GMProbModel(scale_table=None) if entropy_model is None else entropy_model
    
        
    @staticmethod
    def get_hash(tensor: np.ndarray) -> str:
        return hashlib.md5(tensor).hexdigest()            
 
    def encode(self, x, sigma, masks, name=None, entropy_model=None):
        """encode
 
        Args:
            x: shape=[B, H, W, C]
            sigma: shape=[B, H, W, C]
            name:   label for the data
            entropy_model: entropy model of SGM with quantized sigmas
 
        Returns:
            None:
        """                
        model = self.get_model(entropy_model)
        x = self.convert_data(x).to(dtype=torch.int16).cpu().numpy().copy()
        indexes = model.build_indexes(sigma).to(dtype=torch.uint8).cpu().numpy().copy()
        masks = masks.cpu().numpy().copy()
        self.backend.encode_sgm(indexes, x, masks)
 
 
    def decode(self, sigma, masks, name=None, entropy_model=None):
        """decode
 
        Args:
            sigma: shape=[B, H, W, C]
            name:   label for the data
            entropy_model: entropy model of SGM with quantized sigmas
 
        Returns:
            x: shape=[C, H, W]
        """
 
        model = self.get_model(entropy_model)
        indexes = model.build_indexes(sigma).to(dtype=torch.uint8).cpu().numpy()
        x = np.zeros(indexes.shape, dtype=np.int16)
        masks = masks.cpu().numpy()
        self.backend.decode_sgm(indexes, x, masks)
        x = torch.from_numpy(x.astype(np.float32))
        return x
