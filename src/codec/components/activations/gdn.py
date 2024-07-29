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
import torch.nn.functional as functional
from torch.autograd import Function


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size()) * bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0
        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    Based on the papers:

    > "Density modeling of images using a generalized normalization transformation"<br />
    > J. Ballé, V. Laparra, E.P. Simoncelli<br />
    > https://arxiv.org/abs/1511.06281

    > "End-to-end optimized image compression"<br />
    > J. Ballé, V. Laparra, E.P. Simoncelli<br />
    > https://arxiv.org/abs/1611.01704

    Implements an activation function that is essentially a multivariate
    generalization of a particular sigmoid-type function:
    ```
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]^2))
    ```
    where `i` and `j` run over channels. This implementation never sums across
    spatial dimensions. It is similar to local response normalization, but much
    more flexible, as `beta` and `gamma` are trainable parameters.
    """
    def __init__(self,
                 channels,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=0.1,
                 reparam_offset=2**-18):
        super(GDN, self).__init__()
        reparam_offset = torch.FloatTensor([reparam_offset])
        pedestal = reparam_offset**2
        self.inverse = inverse
        self.beta_bound = (beta_min + reparam_offset**2)**0.5
        self.gamma_bound = reparam_offset

        # create beta param
        beta = torch.sqrt(torch.ones(channels) + pedestal)
        self.beta = nn.Parameter(beta)

        # create gamma param
        eye = torch.eye(channels)
        g = gamma_init * eye
        g = g + pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)
        self.pedestal = nn.Parameter(pedestal, requires_grad=False)

    def forward(self, x):
        _, c, _, _ = x.shape

        # beta bound and reparam
        lb1 = LowerBound.apply
        beta = lb1(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal

        # gamma bound and reparam
        lb2 = LowerBound.apply
        gamma = lb2(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma = gamma.view(c, c, 1, 1)

        # norm pool calc
        norm_ = functional.conv2d(x**2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # apply norm
        if self.inverse:
            x = x * norm_
        else:
            x = x / norm_

        return x
