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

from .base_prob_wrapper import WrapperProbBase


class SgtProbWrapper(WrapperProbBase):
    def __init__(self, backend):
        super(SgtProbWrapper, self).__init__(backend)
        self.use_simple_model = True
        self.model_expgolomb_code = True

    def encode(self, x, sigma, mask = None, name=None, entropy_model=None):
        """

        Args:
            x: shape=[H, W, C]
            sigma: shape=[H, W, C]
            max_symbol_value:
            name:
            code_mode:

        Returns:

        """
        # print('>>> Proxy/SgtProbWrapper: encode', flush=True)
        model = GMProbModel(scale_table=None) if entropy_model is None else entropy_model
        model = model.to(x.device)

        freq = self.compute_freq(model, x, sigma)
        bits = self.compute_bits(freq)
        self.backend.update_label_attrs(name, bits, freq)
        # print('<<< Proxy/SgtProbWrapper: encode', flush=True)

    def compute_bits(self, freq):
        bits = -freq.clamp(max=1.0 - 1E-9).log2().sum()
        return bits

    def compute_freq(self, model, x: torch.Tensor, sigma):
        if self.use_simple_model:
            return model(x, scale=sigma)
        else:
            indexes = model.build_indexes(sigma.squeeze(0))
            cdfs = model.quantized_cdf
            cdf_lengths = model.cdf_length.tolist()
            offsets = model.offset
            max_cdf_val = (1 << model.entropy_coder_precision)

            ans = torch.zeros_like(x, dtype=torch.float64)
            for i in range(x.numel()):
                cdf_idx = indexes[i]
                max_val = cdf_lengths[cdf_idx] - 2
                cdf_cur = cdfs[cdf_idx]
                value = x[i] - offsets[cdf_idx]
                old_value = value.item()
                value.clamp_(0, max_val)
                if (old_value != value.item()) and self.model_expgolomb_code:
                    ans[i] = (cdf_cur[max_val + 1] - cdf_cur[max_val]).float() / max_cdf_val
                    raw_value = old_value
                    if old_value < 0:
                        raw_value = -2 * old_value + 1
                    elif old_value > max_val:
                        raw_value = 2 * (old_value - max_val)
                    border_value_prob = ((cdf_cur[max_val + 1] - cdf_cur[max_val]).float() /
                                         max_cdf_val)
                    unary_prob = np.power(0.5, np.log2(np.log2((raw_value + 1) * (raw_value + 1))))
                    ans[i] = border_value_prob.double() * torch.tensor(unary_prob,
                                                                       dtype=torch.float64)

                else:
                    ans[i] = (cdf_cur[value + 1] - cdf_cur[value]).float() / max_cdf_val

            return ans
