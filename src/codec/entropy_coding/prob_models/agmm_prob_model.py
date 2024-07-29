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

import ctypes

import numpy as np
import torch

from .base_prob_model import BaseProbModel, ProbModelTypes


class AgmmProbModel(BaseProbModel):
    """AgmmProbModel
    """
    def __init__(self, mu, sigma_l, sigma_r, weight, symbol_num=512):
        """

        Args:
            mu: shape=[GMM_SIZE, H, W, C].
            sigma_l: shape=[GMM_SIZE, H, W, C].
            sigma_r: shape=[GMM_SIZE, H, W, C].
            weight: shape=[GMM_SIZE, H, W, C].
            symbol_num: symbol's number, default value=512.
        """
        name = ProbModelTypes.set_prob_model('AgmmProbModel')
        super(AgmmProbModel, self).__init__(name, symbol_num)
        self._init_model_params(mu, sigma_l, sigma_r, weight)

    def _init_model_params(self, mu, sigma_l, sigma_r, weight):
        self.params = {
            'mu': mu,
            'sigma_l': sigma_l,
            'sigma_r': sigma_r,
            'weight': weight,
        }

    def to_ctypes(self, dtype=np.float32):
        gmm_size = len(self.params['mu']) if len(self.params['mu'].shape) > 1 else 1
        height, width, chs = self.parse_data_shape(self.params['mu'])

        # shape=[GMM_SIZE * 4, H, W, C] => [4, GMM_SIZE, H, W, C] => [H, W, C, GMM_SIZE, 4]
        params = [
            self.params['mu'], self.params['sigma_l'], self.params['sigma_r'],
            self.params['weight']
        ]
        tensor = torch.cat(params, dim=0)
        tensor = tensor.view(-1, gmm_size, height, width, chs).permute(2, 3, 4, 1, 0).contiguous()
        array = np.asarray(tensor.cpu().numpy(), dtype=dtype)
        cptr = array.ctypes.data_as(ctypes.c_char_p)

        return cptr, gmm_size, height, width, chs

    @staticmethod
    def parse_data_shape(mu):
        x_shape = mu[0].shape if len(mu.shape) > 1 else mu.shape

        if len(x_shape) == 5:
            chs = x_shape[2]
        else:
            chs = x_shape[-1]

        if len(x_shape) == 3:
            height, width, _ = x_shape
        else:
            height, width = 1, 1

        return height, width, chs
