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

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from torch import Tensor

from src.codec.common.utils import is_int_dtype


def lower_bound_fwd(x: Tensor, bound: Tensor) -> Tensor:
    return torch.max(x, bound)


def lower_bound_bwd(x: Tensor, bound: Tensor, grad_output: Tensor):
    pass_through_if = (x >= bound) | (grad_output < 0)
    return pass_through_if * grad_output, None


class LowerBoundFunction(torch.autograd.Function):
    """Autograd function for the `LowerBound` operator."""
    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return lower_bound_fwd(x, bound)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        return lower_bound_bwd(x, bound, grad_output)


class LowerBound(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.

    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """

    bound: Tensor

    def __init__(self, bound: float, device='cpu'):
        super().__init__()
        self.register_buffer('bound', torch.Tensor([float(bound)]).to(device))

    @torch.jit.unused
    def lower_bound(self, x):
        return LowerBoundFunction.apply(x, self.bound)

    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return self.lower_bound(x)


class GMProbModel(nn.Module):
    """Symmetric Single Gaussian Model(SGM) as ProbModel for the conditional entropy model.
    """
    const = 2**(-0.5)  # reciprocal of sqrt(2)
    scale_min = float(1e-9)

    def __init__(self,
                 scale_table: Optional[Union[List, Tuple]],
                 scale_bound: float = 0.11,
                 tail_mass: float = 1e-9,
                 entropy_coder_precision: int = 16,
                 likelihood_bound: float = 1e-9,
                 scale_level: int = 35,
                 scale_max: float = 100,
                 scale_min: float = 0.11,
                 bound_offset: float = 0.5,
                 **kwargs):
        super(GMProbModel, self).__init__(**kwargs)

        if not isinstance(scale_table, (type(None), list, tuple)):
            raise ValueError(f'Invalid type for scale_table "{type(scale_table)}"')

        if isinstance(scale_table, (list, tuple)) and len(scale_table) < 1:
            raise ValueError(f'Invalid scale_table length "{len(scale_table)}"')

        if scale_table and (scale_table != sorted(scale_table) or any(s <= 0
                                                                      for s in scale_table)):
            raise ValueError(f'Invalid scale_table "({scale_table})"')

        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)

        self.tail_mass = float(tail_mass)
        self.scale_level = int(scale_level)
        self.scale_max = float(scale_max)
        self.scale_min = float(scale_min)
        self.bound_offset = float(bound_offset)
        if scale_bound is None and scale_table:
            scale_bound = self.scale_table[0]
        if scale_bound <= 0:
            raise ValueError('Invalid parameters')
        self.lower_bound_scale = LowerBound(scale_bound)
        self.entropy_coder_precision = int(entropy_coder_precision)

        self.register_buffer(
            'scale_table',
            self._prepare_scale_table(scale_table) if scale_table else torch.Tensor(),
        )

        self.register_buffer(
            'scale_bound',
            torch.Tensor([float(scale_bound)]) if scale_bound is not None else None,
        )

        self.register_buffer('_offset', torch.IntTensor())
        self.register_buffer('_quantized_cdf', torch.IntTensor())
        self.register_buffer('_cdf_length', torch.IntTensor())

        self.scale_table_precision = 0
        
        self.update_scale_table(False)
        self._register_load_state_dict_pre_hook(self._load_from_state_dict_hook)
    
    @property    
    def use_tables(self):
        return self.scale_table_precision != 0
        
    def set_param_precision( self, precision ):
        self.scale_table_precision = precision
        self.scale_table_precision_minus_1_exp = 1 << (self.scale_table_precision - 1)

    def get_scale_table(self):
        SCALES_MIN = self.scale_min
        SCALES_MAX = self.scale_max
        SCALES_LEVELS = self.scale_level
        return torch.exp(torch.linspace(math.log(SCALES_MIN), math.log(SCALES_MAX), SCALES_LEVELS))

    def get_scale_bound_table(self):
        SCALES_MIN = self.scale_min
        SCALES_MAX = self.scale_max
        SCALES_LEVELS = self.scale_level
        log_bounds = torch.linspace(math.log(SCALES_MIN), math.log(SCALES_MAX), SCALES_LEVELS)
        if self.bound_offset > 0:
            log_bounds[:-1] = log_bounds[1:] * self.bound_offset + (1-self.bound_offset) * log_bounds[:-1]
        else:
            log_bounds[1:] = log_bounds[:-1] * -self.bound_offset + (1+self.bound_offset) * log_bounds[1:]
        return torch.exp(log_bounds)

    def update_scale_table(self, force=False):
        # Check if we need to update the gaussian conditional parameters, the
        # offsets are only computed and stored when the conditonal model is
        # updated.
        if self._offset.numel() > 0 and not force:
            return False
        device = self.scale_table.device
        scale_table = self.get_scale_table()
        self.scale_table = self._prepare_scale_table(scale_table).to(device)
        self.update()
        scale_bound = self.get_scale_bound_table()
        self.scale_table = self._prepare_scale_table(scale_bound).to(device)
        return True
    
    def _load_from_state_dict_hook(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        names = list()
        for k in state_dict.keys():
            if k.startswith(prefix) and len(k[len(prefix):].split('.')) == 1:
                names.append(k)
        for k in names:
            del state_dict[k]
        if self.scale_table.shape[0] != self.scale_level:
            self.update_scale_table(force=True)

        if self.use_tables:
            self.scale_table = ( self.scale_table * (1 << self.scale_table_precision) ).round().int()
        else:
            print('Warning: GMProbModel is used in floating point mode') 
        state_dict[f"{prefix}scale_table"] = self.scale_table
        state_dict[f"{prefix}scale_bound"] = self.scale_bound
        state_dict[f"{prefix}_offset"] = self._offset
        state_dict[f"{prefix}_quantized_cdf"] = self._quantized_cdf
        state_dict[f"{prefix}_cdf_length"] = self._cdf_length

    @staticmethod
    def get_cdf_matrix(pmf_matrix):
        cdf_matrix = torch.zeros([pmf_matrix.shape[0], pmf_matrix.shape[1] + 1], dtype=torch.int64)
        cdf_matrix[:, 1:] = torch.cumsum(pmf_matrix, dim=1)
        return cdf_matrix

    def update(self):
        multiplier = -self._standardized_quantile(self.tail_mass / 2)
        pmf_center = torch.ceil(self.scale_table * multiplier).int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.abs(torch.arange(max_length, device=device).int() - pmf_center[:, None])
        samples_scale = self.scale_table.unsqueeze(1)
        samples = samples.float()
        samples_scale = samples_scale.float()
        upper = self.get_std_cum_freq((0.5 - samples) / samples_scale)
        lower = self.get_std_cum_freq((-0.5 - samples) / samples_scale)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]
        quantized_pmf = (pmf * ((1 << self.entropy_coder_precision) - 1)).round()
        norm_quantized_pmf = quantized_pmf / torch.sum(quantized_pmf, dim=1).unsqueeze(1)

        quantized_cdf = (torch.cumsum(norm_quantized_pmf, dim=1) * ((1 << self.entropy_coder_precision) - 1)).round().to(dtype=torch.long)
        self._quantized_cdf = quantized_cdf
        self._offset = -pmf_center
        self._cdf_length = pmf_length + 2

    def build_indexes(self, scales: Tensor) -> Tensor:
        indexes = torch.clamp(torch.bitwise_right_shift(scales + self.scale_table_precision_minus_1_exp, self.scale_table_precision), 0, self.scale_level - 1)
        assert (is_int_dtype(indexes))
        return indexes

    def get_std_cum_freq(self, x):
        half = float(0.5)
        const = float(-(2**-0.5))
        cum_freq = half * torch.erfc(const * x)
        return cum_freq

    def _likelihood(self, inputs: Tensor, scales: Tensor, means: Tensor) -> Tensor:
        scales = self.lower_bound_scale( scales / (1 << self.scale_table_precision) )
        values = torch.abs((inputs - means) if means is not None else inputs)
        upper = self.get_std_cum_freq((0.5 - values) / scales)
        lower = self.get_std_cum_freq((-0.5 - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _index_to_scale(self, scale):
        scale = torch.clamp(torch.bitwise_right_shift(scale + self.scale_table_precision_minus_1_exp, self.scale_table_precision), 0, self.scale_level - 1)
        log_k = (np.log(self.scale_max) - np.log(self.scale_min)) / (self.scale_level - 1)
        log_b = np.log(self.scale_min)
        scale = torch.exp(scale * log_k + log_b) * (1 << self.scale_table_precision)
        return scale

    def forward(
        self,
        input: Tensor,
        scale: Tensor,
        mean: Optional[Tensor] = None,
        training: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        # outputs = self.quantize(inputs, "noise" if training else "dequantize", means)
        output = input
        if self.use_tables:
            scale = self._index_to_scale(scale)
        likelihood = self._likelihood(output, scale, mean)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return likelihood

    @staticmethod
    def _prepare_scale_table(scale_table):
        return torch.Tensor(tuple(float(s) for s in scale_table))

    @staticmethod
    def _standardized_quantile(quantile):
        return stats.norm.ppf(quantile)

    @property
    def offset(self):
        return self._offset

    @property
    def quantized_cdf(self):
        return self._quantized_cdf

    @property
    def cdf_length(self):
        return self._cdf_length
