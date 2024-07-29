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
from ..base.utils import generate_dct_filters, generate_lanczos_filters
##
from .params import TableBasedParams


class TableBasedInterpolation(InterpolationFilterBase):
    def __init__(self, *args, **kwargs):
        super(TableBasedInterpolation, self).__init__(*args, **kwargs)
        self._params_table_based = TableBasedParams()

    def build_model(self):

        if self.filter_type == 'dct':
            if self.filter_len != 8 or self.filter_num != 16:
                self.logger.critical(
                    'Invalid filter_len or filter_num for DCT-IF, filter_len should be 8, filter_num should be 16'
                )
                raise NotImplementedError

            self.filters_2D = generate_dct_filters(self.filter_len, self.filter_num, self.device)
        elif self.filter_type == 'lanczos':
            self.filters_2D = generate_lanczos_filters(self.filter_len, self.filter_num,
                                                       self.device)

        else:
            raise NotImplementedError

    def forward(self, x, mv_field):
        if self.parallel:
            return self._warp_table_based_parallel_by_batches(x, mv_field)
        else:
            return self._warp_table_based(x, mv_field)

    def _warp_table_based(self, x, in_mv_field):

        B, C, H_in, W_in = x.size()
        _, _, H, W = in_mv_field.size()

        N = self.filter_len
        N_div_2 = int(N / 2)
        num_filters = self.filter_num

        dev = x.device

        filters_2D = self.filters_2D.to(device=dev)

        indices_x = torch.arange(1 - N_div_2, 1 + N_div_2,
                                 device=dev).repeat(N, 1).expand(H, W, N, N)
        indices_y = torch.arange(1 - N_div_2, 1 + N_div_2,
                                 device=dev).view(N, 1).repeat(1, N).expand(H, W, N, N)
        offsets_x_orig = torch.arange(W, device=dev).repeat(H, 1).expand(H, W).clone()
        offsets_y_orig = torch.arange(H, device=dev).view(H, 1).repeat(1, W).expand(H, W).clone()

        out_img_tensor = torch.empty(B, C, H, W, device=dev)

        for b in range(B):
            mv_field = in_mv_field[b, :, :, :]

            mv_field = ((mv_field * num_filters).round()) / num_filters

            integer_mv_field = torch.floor(mv_field).int()
            filter_indices = torch.round((mv_field - integer_mv_field) * num_filters).long()

            offsets_x = offsets_x_orig + integer_mv_field[0, :, :]
            offsets_y = offsets_y_orig + integer_mv_field[1, :, :]

            del integer_mv_field
            del mv_field

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            expanded_indices_x = indices_x + offsets_x.view(H, W, 1, 1)
            expanded_indices_y = indices_y + offsets_y.view(H, W, 1, 1)

            expanded_indices_x.clamp_(0, W_in - 1)
            expanded_indices_y.clamp_(0, H_in - 1)

            input_img_slice = x[b, :, expanded_indices_y, expanded_indices_x]

            del expanded_indices_x
            del expanded_indices_y

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            filters_2D_for_the_whole_img = filters_2D[filter_indices[0, :, :],
                                                      filter_indices[1, :, :]] / 4096.0
            out_img_tensor[b] = torch.einsum('cijkl,ijkl->cij', input_img_slice,
                                             filters_2D_for_the_whole_img)

            del input_img_slice
            del filters_2D_for_the_whole_img
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mask = torch.ones(x.size(), device=dev)

        return out_img_tensor, mask

    def _warp_table_based_parallel_by_batches(self, x, mv_field):

        B, C, H, W = x.size()

        N = self.filter_len
        N_div_2 = int(N / 2)
        num_filters = self.filter_num

        dev = x.device

        filters_2D = self.filters_2D.to(device=dev)

        indices_x = torch.arange(1 - N_div_2, 1 + N_div_2,
                                 device=dev).repeat(N, 1).expand(B, H, W, N, N)
        indices_y = torch.arange(1 - N_div_2, 1 + N_div_2,
                                 device=dev).view(N, 1).repeat(1, N).expand(B, H, W, N, N)
        offsets_x = torch.arange(W, device=dev).repeat(H, 1).expand(B, H, W).clone()
        offsets_y = torch.arange(H, device=dev).view(H, 1).repeat(1, W).expand(B, H, W).clone()

        mv_field = ((mv_field * num_filters).round()) / num_filters

        integer_mv_field = torch.floor(mv_field).int()
        filter_indices = torch.round((mv_field - integer_mv_field) * num_filters).long()

        offsets_x += integer_mv_field[:, 0, :, :]
        offsets_y += integer_mv_field[:, 1, :, :]

        expanded_indices_x = indices_x + offsets_x.view(B, H, W, 1, 1)
        expanded_indices_y = indices_y + offsets_y.view(B, H, W, 1, 1)

        expanded_indices_b = torch.arange(B, device='cpu').view(B, 1, 1, 1,
                                                                1).expand(B, H, W, N, N).clone()
        expanded_indices_b = expanded_indices_b.to(dev)

        expanded_indices_x[expanded_indices_x < 0] = 0
        expanded_indices_y[expanded_indices_y < 0] = 0
        expanded_indices_x[expanded_indices_x > (W - 1)] = W - 1
        expanded_indices_y[expanded_indices_y > (H - 1)] = H - 1

        input_img_slice = x[expanded_indices_b, :, expanded_indices_y, expanded_indices_x]
        input_img_slice = input_img_slice.permute(0, 5, 1, 2, 3, 4)
        filters_2D_for_the_whole_img = filters_2D[filter_indices[:, 0, :, :],
                                                  filter_indices[:, 1, :, :]] / 4096.0
        out_img_tensor = torch.einsum('bcijkl,bijkl->bcij', input_img_slice,
                                      filters_2D_for_the_whole_img)

        mask = torch.ones(x.size(), device=dev)

        return out_img_tensor, mask
