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

##
import torch

from ...prob_models import BypassProbModel
from .base_prob_wrapper import WrapperProbBase


class BypassProbWrapper(WrapperProbBase):
    def __init__(self, backend):
        super(BypassProbWrapper, self).__init__(backend)

    def encode(self, x, max_symbol_value=1, name=None):
        """encode

        Args:
            x: shape = [B, C, H, W], [C, H, W], or flattened variants
            max_symbol_value:
            name:

        Returns:
            none:
        """
        # print('>>> Proxy/BypassProbWrapper: encode', flush=True)
        prob_model = BypassProbModel(symbol_num=(max_symbol_value + 1))

        bits = self.compute_bits(x, prob_model)
        freq = self.compute_freq(x)
        self.backend.update_label_attrs(name, bits, freq)
        # print('<<< Proxy/BypassProbWrapper: encode', flush=True)

    def compute_bits(self, x, prob_model):
        x = self.convert_data(x)
        size = self.get_size(x)
        bits_per = self.compute_symbol_bits(prob_model.symbol_num, x.device, x.dtype)
        bits = (size * bits_per).sum()
        return bits

    def compute_freq(self, x):
        x = self.convert_data(x)
        freq = 0.5 * torch.ones(x.shape, device=x.device)
        return freq
