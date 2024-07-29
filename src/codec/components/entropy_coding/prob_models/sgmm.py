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

from .ssgm import SSGMProbModel


class SGMMProbModel(SSGMProbModel):
    """Symmetric Gaussian Mixture Model(GMM) as ProbModel for the conditional entropy model.
    """
    def __init__(self, scale_bound=1e-9, freq_bound=1e-9):
        super(SGMMProbModel, self).__init__(scale_bound=scale_bound, freq_bound=freq_bound)

    def forward(self, x, mean, scale, weight):
        scale = scale.clamp(min=self.scale_bound)
        freq = self.get_freq(x, mean, scale, weight)
        freq = freq.clamp(min=self.freq_bound, max=1)
        return freq

    def get_freq(self, x, mean, scale, weight):
        upper = self.norm_data(x + 0.5, mean, scale)
        lower = self.norm_data(x - 0.5, mean, scale)
        upper_cum_freq = self.get_std_cum_freq(upper)
        lower_cum_freq = self.get_std_cum_freq(lower)
        freq = upper_cum_freq - lower_cum_freq
        freq = torch.sum(weight * freq, dim=0)
        return freq
