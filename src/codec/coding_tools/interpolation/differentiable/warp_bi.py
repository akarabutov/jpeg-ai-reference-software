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

##
from ..base.interface import InterpolationFilterBase


class BiInterpolation(InterpolationFilterBase):
    def forward(self, x, mv_field):
        return BiInterpolation.warp_bi(x, mv_field, mode='bilinear')

    @staticmethod
    def warp_bi(x, flo, mode, use_mask=False):
        """ Warp an image/tensor (im2) back to im1, according to the optical flow.
        This method is used from the general differentiable interpolation, so it's more convenient to have is static.

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        device = x.device
        B, _, H_in, W_in = x.size()
        _, _, H, W = flo.size()
        # mesh grid
        xx = torch.arange(0, W, device=device, dtype=torch.float).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device=device, dtype=torch.float).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), dim=1)

        vgrid = grid + flo

        # scale grid to [-1,1]
        vgrid = 2.0 * vgrid / torch.tensor(
            [max(W_in - 1, 1), max(H_in - 1, 1)], device=device,
            dtype=torch.float).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = torch.nn.functional.grid_sample(x,
                                                 vgrid,
                                                 mode=mode,
                                                 align_corners=True,
                                                 padding_mode='border')

        if use_mask:
            mask = torch.ones(x.size(), device=device)
            mask = torch.nn.functional.grid_sample(mask,
                                                   vgrid,
                                                   mode='nearest',
                                                   align_corners=True,
                                                   padding_mode='zeros')

            mask[mask < 0.9999] = 0
            mask[mask > 0] = 1

            return output, mask
        else:
            return output, None
