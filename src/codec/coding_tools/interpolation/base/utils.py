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

import torch


# ############## Lanczos filters ##########################
def sinc(x):
    x = x + 1e-9

    ans = torch.sin(math.pi * x) / (math.pi * x)
    return ans


def gen_lanczos_filters_for_fractional_offsets(mv_field_frac, T, band_correction_xy=[1, 1]):
    # x,y from [0, 1), T -- num taps in filter

    if not isinstance(band_correction_xy, torch.Tensor):
        band_correction_xy = torch.tensor(band_correction_xy, device=mv_field_frac.device)

    band_correction_xy = band_correction_xy.to(device=mv_field_frac.device).view(1, 2, 1, 1)

    window_size_xy = (T / 2) * band_correction_xy

    B, _, H, W = mv_field_frac.size()
    filters = torch.zeros(B, T * T, H, W, device=mv_field_frac.device)
    offset_list = torch.arange(-T // 2 + 1, T // 2 + 1)

    for i, y_offset in enumerate(offset_list):
        for j, x_offset in enumerate(offset_list):
            offset_pair = torch.tensor([x_offset, y_offset],
                                       device=mv_field_frac.device).view(1, 2, 1, 1)
            arg = -mv_field_frac + offset_pair
            arg2 = arg / window_size_xy
            filters[:, i * T + j, :, :] = sinc(arg[:, 0, :, :]) * sinc(arg2[:, 0, :, :]) * sinc(
                arg[:, 1, :, :]) * sinc(arg2[:, 1, :, :])
    return filters / filters.sum(dim=1).unsqueeze(-3)


def generate_lanczos_filters(filter_len, num_filters, target_device, band_correction_xy=[1, 1]):

    offsets = torch.arange(0, num_filters, dtype=torch.float, device=target_device) / num_filters

    offsets_2D = torch.zeros(2, num_filters, num_filters, device=target_device)
    offsets_2D[1] = offsets.repeat(num_filters, 1)
    offsets_2D[0] = offsets.view(num_filters, 1).repeat(1, num_filters)

    filters_2D = gen_lanczos_filters_for_fractional_offsets(offsets_2D.unsqueeze(0), filter_len,
                                                            band_correction_xy)
    filters_2D = filters_2D.squeeze().permute(1, 2, 0).view(num_filters, num_filters, filter_len,
                                                            filter_len) * 4096

    return filters_2D


# ##################### DCT-IF Filters #####################
filters_8tap_dct_16 = [[0, 0, 0, 64, 0, 0, 0, 0], [0, 1, -3, 63, 4, -2, 1, 0],
                       [-1, 2, -5, 62, 8, -3, 1, 0], [-1, 3, -8, 60, 13, -4, 1, 0],
                       [-1, 4, -10, 58, 17, -5, 1, 0], [-1, 4, -11, 52, 26, -8, 3, -1],
                       [-1, 3, -9, 47, 31, -10, 4, -1], [-1, 4, -11, 45, 34, -10, 4, -1],
                       [-1, 4, -11, 40, 40, -11, 4, -1], [-1, 4, -10, 34, 45, -11, 4, -1],
                       [-1, 4, -10, 31, 47, -9, 3, -1], [-1, 3, -8, 26, 52, -11, 4, -1],
                       [0, 1, -5, 17, 58, -10, 4, -1], [0, 1, -4, 13, 60, -8, 3, -1],
                       [0, 1, -3, 8, 62, -5, 2, -1], [0, 1, -2, 4, 63, -3, 1, 0]]

filters_8tap_dct_4 = [[0, 0, 0, 64, 0, 0, 0, 0], [-1, 4, -10, 58, 17, -5, 1, 0],
                      [-1, 4, -11, 40, 40, -11, 4, -1], [0, 1, -5, 17, 58, -10, 4, -1]]


def generate_dct_filters(filter_len, num_filters, target_device, to_print=False):

    if num_filters == 4:
        filters_1D = filters_8tap_dct_4
    elif num_filters == 16:
        filters_1D = filters_8tap_dct_16
    else:
        raise NotImplementedError

    filters_2D = []

    for x in range(num_filters):
        filters_2D.append([])
        for y in range(num_filters):
            filters_2D[x].append([filters_1D[x].copy() for i in range(filter_len)])

            for i in range(filter_len):
                for j in range(filter_len):
                    filters_2D[x][y][i][j] *= filters_1D[y][i]

    if to_print:
        for x in range(num_filters):
            for y in range(num_filters):
                print('filter dx = {}, dy = {} :'.format(x, y))
                for i in range(filter_len):
                    line = filters_2D[x][y][i]
                    print(' '.join(['{:>8}' for i in range(filter_len)]).format(*line))

    return torch.tensor(filters_2D, device=target_device)
