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
import torch.nn.functional as F
import numpy as np
from src.codec.components.entropy_coding import FactorizedProbModel
from .base_prob_wrapper import BaseProbWrapper

class CustomProbWrapper(BaseProbWrapper):
    def __init__(self, backend):
        super(CustomProbWrapper, self).__init__(backend)

    def encode(self, x, model: FactorizedProbModel, max_symbol_value=512, mean=None, name=None):
        """
        Args:
            x: shape=[B, C, H, W]
            model:
            max_symbol_value:
            mean:
            name:

        Returns:
            None:
        """
        self.check_range(x, min=0, max=max_symbol_value)
        chs, _, _ = self.parse_data_shape(x)
        pmfs_z_tensor = model.get_freq_table(chs, max_symbol_value + 1, mean)
        pmfs_z_norm_np = self.normalize_z(pmfs_z_tensor)
        x = x.squeeze(0).byte().cpu().numpy()
        x = x.reshape(x.shape[0], -1)
        self.backend.encode_factorize(pmfs_z_norm_np, x.copy())
        
 
    def decode(self,
               shape,
               model: FactorizedProbModel,
               max_symbol_value=512,
               mean=None,
               name=None):
        """
 
        Args:
            shape:
            model:
            max_symbol_value:
            mean:
            name:
 
        Returns:
            x: shape=[B, C, H, W]
        """
        chs = shape[-3]
        pmfs_z_tensor = model.get_freq_table(chs, max_symbol_value + 1, mean)
        pmfs_z_norm_np = self.normalize_z(pmfs_z_tensor)
        z = np.zeros([chs, shape[-2] * shape[-1]], dtype=np.int8)
        self.backend.decode_factorize(pmfs_z_norm_np, z)
        z = torch.from_numpy(z.astype(np.int8).reshape(shape))
        return z

    # ##################################################################################################################
    #  service methods
    # ##################################################################################################################
    @staticmethod
    def normalize_z(pmfs_z_tensor):
        pmfs_z_cumsum = np.cumsum(pmfs_z_tensor.cpu().numpy(), axis=1)
        pmfs_z_norm = ((pmfs_z_cumsum * 255 + (pmfs_z_cumsum[:, -1:] >> 1)) // pmfs_z_cumsum[:, -1:]).astype(np.uint8)
        return pmfs_z_norm
 
    @staticmethod
    def parse_data_shape(x):
        length = len(x.shape)
 
        if length == 4:
            _, chs, height, width = x.shape
        elif length == 3:
            chs, height, width = x.shape
        elif length == 1:
            chs, height, width = x.shape[0], 1, 1
        else:
            raise NotImplementedError
 
        return chs, height, width
