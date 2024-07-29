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


class AsgmProbModel(BaseProbModel):
    """AsgmProbModel
    """
    def __init__(self, mu, sigma_l, sigma_r, symbol_num=512):
        """

        Args:
            mu: shape=[H, W, C].
            sigma_l: shape=[H, W, C].
            sigma_r: shape=[H, W, C].
            symbol_num: symbol's number, default value=512.
        """
        name = ProbModelTypes.set_prob_model('AsgmProbModel')
        super(AsgmProbModel, self).__init__(name, symbol_num)
        self._init_model_params(mu, sigma_l, sigma_r)

    def _init_model_params(self, mu, sigma_l, sigma_r):
        self.params = {
            'mu': mu,
            'sigma_l': sigma_l,
            'sigma_r': sigma_r,
        }

    def to_ctypes(self, dtype=np.float32):
        mu = self.params['mu']
        sigma_l = self.params['sigma_l']
        sigma_r = self.params['sigma_r']
        x_size = np.prod(mu.shape).astype(np.uint32)

        mu_cptr = self.tensor_to_cptr(mu, x_size, dtype=np.float32)
        scale_l_cptr = self.tensor_to_cptr(sigma_l, x_size, dtype=np.float32)
        scale_r_cptr = self.tensor_to_cptr(sigma_r, x_size, dtype=np.float32)

        return mu_cptr, scale_l_cptr, scale_r_cptr

    @staticmethod
    def parse_shape(mu):
        x_shape = mu[0].shape

        if len(x_shape) == 4:
            chs = x_shape[1]
        else:
            chs = x_shape[-1]

        if len(x_shape) == 3:
            height, width, _ = x_shape
        else:
            height, width = 1, 1

        return height, width, chs

    @staticmethod
    def tensor_to_cptr(tensor, size, dtype=np.float32):
        ndarr = tensor.cpu().numpy().reshape(size)
        array = np.asarray(ndarr, dtype=dtype)
        cptr = array.ctypes.data_as(ctypes.c_char_p)
        return cptr
