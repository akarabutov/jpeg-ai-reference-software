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
import torch.nn as nn
import torch.nn.functional as functional


class FactorizedProbModel(nn.Module):
    """Factorized Entropy bottleneck.

    The module implements a flexible probability density model to estimate entropy
    of its input tensor, which is described in the appendix of the paper:

    > "Variational image compression with a scale hyperprior" <br />
    > J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston <br />
    > https://arxiv.org/abs/1802.01436

    The module trains an independent probability density model for each channel, but
    assumes that across all other dimensions, the x are i.i.d. (independent
    and identically distributed).

    Because data compression always involves discretization, the outputs of the
    layer are generally only approximations of its x. During training,
    discretization is modeled using additive uniform noise to ensure differentiability.
    The entropies computed during training are differential entropies. During evaluation,
    the data is actually quantized, and the entropies are discrete (Shannon entropies).

    Args:
        init_scale: A float scaling factor determining the initial width of the
            probability densities. This should be chosen big enough so that the
            range of values of the layer x roughly falls within the interval
            [`-init_scale`, `init_scale`] at the beginning of training.
        filters: An iterable of ints, giving the number of filters at each layer of
            the density model. Generally, the more filters and layers, the more
            expressive is the density model in terms of modeling more complicated
            distributions of the layer x. For details, refer to the paper
            referenced above. The default is `[3, 3, 3]`, which should be sufficient
            for most practical purposes.
        tail_mass: A float between 0 and 1. The bottleneck layer automatically
            determines the range of input values based on their frequency of
            occurrence. Values occurring in the tails of the distributions will not be
            encoded with range coding, but using a Golomb-like code. `tail_mass`
            determines the amount of probability mass in the tails which will be
            Golomb-coded. For example, the default value of `2 ** -8` means that on
            average, one 256th of all values will use the Golomb code.
    """
    def __init__(self,
                 channels=192,
                 init_scale=10,
                 filters=(3, 3, 3),
                 tail_mass=2**-8,
                 freq_bound=1e-9,
                 **kwargs):
        super(FactorizedProbModel, self).__init__()
        # basic backup
        self.init_scale = float(init_scale)
        self.filters = filters
        self.tail_mass = tail_mass
        self.freq_bound = freq_bound
        # create cdf parameters
        expand_filters = (1, ) + self.filters + (1, )

        max_symbol = kwargs.get('max_symbol', 0)

        self.register_buffer('freqs_int', torch.zeros(channels, max_symbol + 1, dtype=torch.int32))
        self.register_buffer('is_quantized', torch.tensor([False], dtype=torch.bool))

        self.matrices = []
        self.biases = []
        self.factors = []
        for i in range(len(self.filters) + 1):
            # `H` in the paper
            matrix = nn.Parameter(
                torch.FloatTensor(channels, expand_filters[i + 1], expand_filters[i]))
            scale = self.init_scale**(1 / (len(self.filters) + 1))
            init = np.log(np.expm1(1 / scale / expand_filters[i + 1]))
            nn.init.constant_(matrix, init)
            self.matrices.append(matrix)

            # `b` in the paper
            bias = nn.Parameter(torch.FloatTensor(channels, expand_filters[i + 1], 1))
            nn.init.uniform_(bias, a=-0.5, b=0.5)
            self.biases.append(bias)

            # `a` in the paper
            if i < len(self.filters):
                factor = nn.Parameter(torch.FloatTensor(channels, expand_filters[i + 1], 1))
                nn.init.constant_(factor, 0)
                self.factors.append(factor)

        self.matrices = nn.ParameterList(self.matrices)
        self.biases = nn.ParameterList(self.biases)
        self.factors = nn.ParameterList(self.factors)

    def forward(self, x):
        freq = self.get_freq(x)
        freq = freq.clamp(min=self.freq_bound, max=1)
        return freq

    def get_freq_table(self, chs, symbol_num, mean, factor=1E9, init=False):
        if self.is_quantized:
            return self.freqs_int
        else:
            symbols = torch.arange(symbol_num, device=next(self.parameters()).device)
            symbols = symbols.view(1, 1, symbol_num) - mean
            symbols = symbols.repeat(chs, 1, 1).float()

            freqs = self.get_freq(symbols)

            ret = (freqs * factor).int() + 1

            return ret.squeeze(1)

    def get_freq(self, x):
        """Get the likelihood of quantized tensor based on CDF (Cumulative Distribution Function).
        We can use the special rule below to only compute differences in the left tail of the sigmoid.
        This increases numerical stability: sigmoid(x) is 1 for large x, 0 for small x.
        Subtracting two numbers close to 0 can be done with much higher precision than that of close to 1.
        Args:
            x: a tensor with shape=[N, C, H, W]

        Returns:
            freq: w.r.t. the quantized tensor
        """

        if len(x.shape) > 3:
            x, shape = self.reshape_data(x)
        else:
            shape = x.shape
        upper_cum_logit = self.get_cum_logits(x + 0.5, stop_gradient=False)
        lower_cum_logit = self.get_cum_logits(x - 0.5, stop_gradient=False)

        # flip signs if we can move more towards the left tail of the sigmoid
        sign = -(upper_cum_logit + lower_cum_logit).sign().detach()
        upper_cum_freq = (sign * upper_cum_logit).sigmoid()
        lower_cum_freq = (sign * lower_cum_logit).sigmoid()
        freq = torch.abs(upper_cum_freq - lower_cum_freq)

        if shape != x.shape:
            freq = self.reshape_freq(freq, shape)
        return freq

    def get_cum_logits(self, x, stop_gradient):
        if x.is_cuda:
            return self.get_cum_logits_dev(x, stop_gradient)
        else:
            return self.get_cum_logits_cpu(x, stop_gradient)

    def get_cum_logits_cpu(self, x, stop_gradient):
        """Get the cumulative logits before sigmoid.
        Args:
            x: a tensor with shape=[C, 1, NHW]
            stop_gradient:
        Returns:
            logits: a tensor with the same shape of x
        """
        logits = x

        for i in range(len(self.filters) + 1):
            matrix = functional.softplus(self.matrices[i].cpu()).to(self.matrices[i].device)
            if stop_gradient:
                matrix = matrix.detach()

            logits = torch.bmm(matrix.cpu(), logits.cpu()).to(logits.device)

            bias = self.biases[i]
            if stop_gradient:
                bias = bias.detach()

            logits += bias

            if i < len(self.factors):
                factor = self.factors[i].cpu().tanh().to(self.factors[i].device)
                if stop_gradient:
                    factor = factor.detach()
                logits += factor * logits.cpu().tanh().to(logits.device)

        return logits

    def get_cum_logits_dev(self, x, stop_gradient):
        """Get the cumulative logits before sigmoid.
        Args:
            x: a tensor => (C, 1, NHW)
        Returns:
            logits: a tensor with the same shape of x
        """
        logits = x
        for i in range(len(self.filters) + 1):
            matrix = functional.softplus(self.matrices[i])
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.bmm(matrix, logits)
            bias = self.biases[i]
            if stop_gradient:
                bias = bias.detach()
            logits += bias
            if i < len(self.factors):
                factor = self.factors[i].tanh()
                if stop_gradient:
                    factor = factor.detach()
                logits += factor * logits.tanh()
        return logits

    @staticmethod
    def reshape_data(x):
        """Reshape to [C, 1, NHW] format by commuting channels to front and then collapse.

        Args:
            x:

        Returns:
            y, shape:
        """
        x = x.transpose(0, 1)  # [N, C, H, W] => [C, N, H, W]
        shape = x.shape
        y = x.contiguous()
        y = y.view(shape[0], 1, -1)  # [C, N, H, W] => [C, 1, NHW]
        return y, shape

    @staticmethod
    def reshape_freq(freq, shape):
        """Reshape back to input tensor shape

        Args:
            freq:
            shape: input tensor shape

        Returns:
            freq:
        """
        freq = freq.view(shape)
        freq = freq.transpose(0, 1)
        return freq

    def state_dict(self, destination=None, prefix='', keep_vars=False):

        chs = self.freqs_int.shape[0]
        symbol_num = self.freqs_int.shape[1]

        self.freqs_int.data = self.get_freq_table(chs, symbol_num, symbol_num // 2)

        return super(FactorizedProbModel, self).state_dict(destination, prefix, keep_vars)
