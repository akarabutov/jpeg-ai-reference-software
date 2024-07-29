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

from ..base.interface import InterpolationFilterBase
from ..base.utils import gen_lanczos_filters_for_fractional_offsets
##
from .params import DifFilterParams
from .warp_bi import BiInterpolation


class DifferentiableInterpolation(InterpolationFilterBase):
    def __init__(self, *args, **kwargs):
        super(DifferentiableInterpolation, self).__init__(*args, **kwargs)
        self._params_table_based = DifFilterParams()

    def forward(self, x, mv_field, band_correction_xy=[1, 1]):
        return self._warp_differentiable(x, mv_field, band_correction_xy)

    def build_model(self):

        if self.filter_type == 'lanczos':
            self._gen_filters_for_fractional_offsets = gen_lanczos_filters_for_fractional_offsets
        else:
            raise NotImplementedError

    def _warp_differentiable(self, x, mv_field, band_correction_xy):

        B, _, _, _ = x.size()

        T = self.filter_len

        integer_mv_field = mv_field.floor()  # integer_mv_field.shape = (B, 2, h, w)

        mv_field_frac = mv_field - integer_mv_field

        filters = self._gen_filters_for_fractional_offsets(
            mv_field_frac, T, band_correction_xy)  #filters.shape = (B, T*T, h, w)
        offsets = torch.arange(-T // 2 + 1, T // 2 + 1, dtype=torch.float,
                               device=x.device)  # offsets.shape == (8)

        offsets_2D = torch.zeros(2, T, T, device=x.device)
        offsets_2D[0] = offsets.repeat(T, 1)  #offsets_x
        offsets_2D[1] = offsets.view(T, 1).repeat(1, T)
        offsets_2D = offsets_2D.view(2, T**2).permute(1, 0).unsqueeze(-1).unsqueeze(-1)

        offsets_2D = torch.cat([offsets_2D] * B, 0)  # offsets.shape == (64 * B, 2, 1, 1)
        fields = torch.cat([integer_mv_field] *
                           (T**2), dim=0) + offsets_2D  # fields.shape = (64 * B, 2, h, w)
        refs = torch.cat([x] * (T**2), dim=0)  # refs.shape = (64 * B, c, h, w)

        if fields.numel(
        ) < self.min_num_elements_for_slicing:  #slicing is not needed for small input
            preds, _ = BiInterpolation.warp_bi(refs, fields, mode='nearest',
                                               use_mask=False)  # preds.shape = (64 * B, c, h, w)
            preds = preds.view(B, T**2, preds.shape[-3], preds.shape[-2], preds.shape[-1])
            pred = torch.einsum('bfcij,bfij->bcij', preds, filters)  # pred.shape = (B, c, h, w)
        else:
            _, _, H, _ = fields.shape
            N = self.num_slices

            pred_slices = []
            Hi = (H + N - 1) // N

            vertical_offsets = torch.arange(0, H, step=Hi, device=fields.device).view(N, 1).repeat(
                1, Hi).view(N * Hi)[:H]
            vertical_mv_offsets = torch.stack(
                (torch.zeros(H, dtype=int,
                             device=fields.device), vertical_offsets)).view(1, 2, H, 1)
            fields = fields + vertical_mv_offsets

            for i in range(N):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                preds, _ = BiInterpolation.warp_bi(refs,
                                                   fields[:, :, i * Hi:(i + 1) * Hi],
                                                   mode='nearest',
                                                   use_mask=False)
                preds = preds.view(B, T**2, preds.shape[-3], preds.shape[-2], preds.shape[-1])
                pred = torch.einsum('bfcij,bfij->bcij', preds,
                                    filters[:, :,
                                            i * Hi:(i + 1) * Hi])  # pred.shape = (B, c, h, w)

                pred_slices.append(pred)

                del preds

            pred = torch.cat(tuple(pred_slices), dim=2)

            del filters
            del fields

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mask = torch.tensor(1)

        return pred, mask
