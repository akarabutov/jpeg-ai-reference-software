# The copyright in this software is being made available under the BSD
# License, included below. This software may be subject to other third party
# and contributor rights, including patent rights, and no such rights are
# granted under this license.
#
# Copyright (c) 2010-2026, ITU/ISO/IEC
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


class QuantModule(nn.Module):
    """Quantization (rounding) with the straight-through estimator of the gradient."""

    @staticmethod
    def round_ste(y: torch.Tensor) -> torch.Tensor:
        """Round the tensor; the gradient is passed as is."""
        y_no_grad = y.detach()
        return y_no_grad.round() - y_no_grad + y

    @staticmethod
    def noise_ste(y: torch.Tensor) -> torch.Tensor:
        """Add the rounding error of the tensor shifted by uniform noise (the model of the quantization
        used to estimate likelihoods in training); the gradient is passed as is."""
        y_no_grad_tilde = y.detach() + torch.empty_like(y).uniform_(-0.5, 0.5)
        return y_no_grad_tilde.round() - y_no_grad_tilde + y

    @staticmethod
    def round_masked_ste(data: torch.Tensor, mask: torch.Tensor, int_dtype: torch.dtype) -> torch.Tensor:
        """Quantize a (scaled) residual: zero the skipped elements, clip to the range of an integer
        type and round; the gradient is passed as is.

        Args:
            data (torch.Tensor): residual, the skipped elements are zeroed in-place
            mask (torch.Tensor): elements with False are skipped; None - nothing is skipped
            int_dtype (torch.dtype): integer type the quantized residual has to fit to

        Returns:
            torch.Tensor: quantized residual
        """
        if mask is not None:
            data[~mask.to(torch.bool)] = 0
        type_info = torch.iinfo(int_dtype)
        data = torch.clamp(data, type_info.min, type_info.max)
        return data - (data - data.round()).detach()

    def forward(self, y):
        if self.training:
            return self.round_ste(y), self.noise_ste(y)
        else:
            y_hat = self.round_ste(y)
            return y_hat, y_hat
