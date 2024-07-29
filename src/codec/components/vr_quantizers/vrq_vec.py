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
import numpy as np


class VrqVec(nn.Module):
    tiny_val = 1e-9

    def __init__(self, chs=192, qp_num=6, log_k=(np.log(54.82)-np.log(0.11))/31):
        super(VrqVec, self).__init__()
        self.c = torch.nn.Parameter(torch.ones(
            chs, qp_num))  # vectored parameter denotes multiple channels.
        self.qp_num = qp_num
        self.log_k = log_k
        self._register_load_state_dict_pre_hook(self._hook_load_from_state_dict)

    def forward(self, x, qp_idx, ft=0):
        return self.quantize(x, qp_idx, ft)

    def quantize(self, x, qp_idx, ft=0):
        coefficient = self.get_coefficient(qp_idx, ft)
        alpha = self.unsqueeze_coefficient(coefficient)
        y = x * alpha
        return y
    
    def state_dict(self, *args, **kwargs):
        tmp_c = self.c.clone()
        self.c.data = torch.log(self.c.abs()) / self.log_k
        ans = super().state_dict( *args, **kwargs)
        self.c.data = tmp_c
        return ans
    
    def _hook_load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        state_dict[f'{prefix}c'] = torch.exp(state_dict[f'{prefix}c'] * self.log_k)



    def dequantize(self, x, qp_idx, ft=0):
        coefficient = self.get_coefficient(qp_idx, ft)
        alpha = self.unsqueeze_coefficient(coefficient)
        y = x / (alpha + VrqVec.tiny_val)
        return y

    def get_coefficient(self, qp_idx, interp=0):
        if interp == 0:
            ans = torch.abs(self.c[:, qp_idx])
        elif interp < 0:
            ans = torch.abs(self.c[:, qp_idx]) * (-interp)  # Extreme case, should not happen often
        elif interp >= 1:
            ans = torch.abs(self.c[:, qp_idx]) * (+interp)  # Extreme case, should not happen often
        else:
            if (qp_idx + 1) >= self.qp_num:
                raise AssertionError('Invalid qp_idx={}'.format(qp_idx))

            lhs = torch.abs(self.c[:, qp_idx + 0])
            rhs = torch.abs(self.c[:, qp_idx + 1])
            ans = (lhs**(1 - interp)) * (rhs**interp)
        return ans

    @staticmethod
    def unsqueeze_coefficient(coefficient):
        """Unsqueeze coefficient to match the date shape.

        Args:
            coefficient: shape=[chs,]

        Returns:
            alpha: shape=[1, chs, 1, 1]
        """
        alpha = coefficient.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return alpha
