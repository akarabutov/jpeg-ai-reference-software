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

import torch.nn as nn


class SymmetricProbModel(nn.Module):
    """Symmetric conditional entropy model.
    Args:
        freq_bound: Float. If positive, the returned likelihood values are
            ensured to be greater than or equal to this value. This prevents very
            large gradients with a tyimgal entropy loss (defaults to 1e-9).
    """
    def __init__(self, scale_bound=1e-9, freq_bound=1e-9):
        super(SymmetricProbModel, self).__init__()
        self.scale_bound = scale_bound
        self.freq_bound = freq_bound

    def forward(self, x, mean, scale):
        scale = scale.clamp(min=self.scale_bound)
        freq = self.get_freq(x, mean, scale)
        freq = freq.clamp(min=self.freq_bound, max=1.0)
        return freq

    def get_freq(self, x, mean, scale):
        """This assumes that the standardized cumulative has the property 1 - c(x) = c(-x),
        which means we can compute differences equivalently in the left or right tail of the cumulative.
        The point is to only compute differences in the left tail. This increases numerical stability:
        c(x) is 1 for large x, 0 for small x. Subtracting two numbers close to 0 can be done with
        much higher precision than subtracting two numbers close to 1.

        Args:
            x:
            mean:
            scale:

        Returns:
            freq:
        """
        values = abs(x - mean)
        upper_cum_freq = self.get_std_cum_freq((0.5 - values) / scale)
        lower_cum_freq = self.get_std_cum_freq((-0.5 - values) / scale)
        freq = upper_cum_freq - lower_cum_freq
        return freq

    def get_std_cum_freq(self, x):
        """Evaluate the standardized cumulative density.

        This function should be optimized to give the best possible numerical
        accuracy for negative input values.

        Args:
            x: `Tensor`. The values at which to evaluate the cumulative density.
        Returns:
            A `Tensor` of the same shape as `inputs`, containing the cumulative
            density evaluated at the given inputs.
        """
        raise NotImplementedError('Must inherit from SymmetricProbModel.')
