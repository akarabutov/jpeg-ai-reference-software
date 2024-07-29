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

from src.codec.components.entropy_coding import FactorizedProbModel

from ...prob_models import CustomProbModel
from .base_prob_wrapper import WrapperProbBase


class CustomProbWrapper(WrapperProbBase):
    def __init__(self, backend):
        super(CustomProbWrapper, self).__init__(backend)

    def encode(self, x, model: FactorizedProbModel, max_symbol_value=512, mean=None, name=None):
        """

        Args:
            x: shape=[C, H, W]
            model:
            max_symbol_value:
            mean:
            name:

        Returns:
            none:

        """
        # print('>>> Proxy/CustomProbWrapper: encode', flush=True)
        prob_model = CustomProbModel(model, mean, (max_symbol_value + 1))

        freq = self.compute_freq(x, prob_model)
        bits = self.compute_bits(freq)
        self.backend.update_label_attrs(name, bits, freq)
        # print('<<< Proxy/CustomProbWrapper: encode', flush=True)

    def compute_bits(self, freq):
        bits = -freq.log2().sum()
        return bits

    def compute_freq(self, x, prob_model):
        model = prob_model.get_model_params(key='model')
        mean = prob_model.get_model_params(key='mean')
        freq = model(x - mean)
        return freq
