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
import ctypes

import numpy as np

from src.codec.components.entropy_coding import FactorizedProbModel

from .base_prob_model import BaseProbModel, ProbModelTypes


class CustomProbModel(BaseProbModel):
    """CustomProbModel
    """
    def __init__(self, model: FactorizedProbModel, mean=None, symbol_num=512):
        name = ProbModelTypes.set_prob_model('CustomProbModel')
        super(CustomProbModel, self).__init__(name, symbol_num)
        self._init_model_params(model, mean)

    def _init_model_params(self, model, mean):
        self.params = {
            'model': model,
            'mean': self._set_mean(mean),
        }

    def to_ctypes(self, chs):
        model = self.params['model']
        mean = self.params['mean']
        symbol_num = self.symbol_num

        freqs = model.get_freq_table(chs, symbol_num, mean)

        ndarr = freqs.squeeze(1).cpu().numpy()
        array = np.asarray(ndarr, dtype=np.int32)
        p_cptr = array.ctypes.data_as(ctypes.c_char_p)
        p_size = chs * symbol_num

        return p_cptr, p_size

    # ##################################################################################################################
    #  service methods
    # ##################################################################################################################
    def _set_mean(self, mean):
        if mean is None:
            return self.symbol_num // 2
        elif self.symbol_min <= mean <= self.symbol_max:
            return mean
        else:
            raise ValueError('mean={} not in [{}, {}]'.format(mean, self.symbol_min,
                                                              self.symbol_max))
