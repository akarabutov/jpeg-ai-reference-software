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

from src.codec.common.colorspace import rgb_to_yuv


def get_psnr(img1, img2, cast_to_int=False):
    """Get PSNR through custom logic.
    Args:
        img1 (N3HW): original image
        img2 (N3HW): reconstructed image
        cast_to_int: for evaluating
    Returns:
        a tensor (the average psnr value)
    """
    if cast_to_int:
        img1 = img1.int()
        img2 = img2.int()
    error = (img1 - img2) * (img1 - img2)
    mse_per_image = torch.mean(error.float(), dim=(1, 2, 3))
    psnr_per_image = 10 * torch.log10(255 * 255 / mse_per_image)
    psnr = torch.mean(psnr_per_image)
    return psnr


class PSNR(nn.Module):
    """Wrap for PSNR metric.
    """
    def __init__(self, cast_to_int=False):
        super(PSNR, self).__init__()
        self.cast_to_int = cast_to_int

    def forward(self, img1, img2):
        return 100 - get_psnr(img1, img2, self.cast_to_int)


def get_yuv_psnr(img1, img2, cast_to_int=False):
    """Get PSNR through custom logic.
    Args:
        img1 (N3HW): original image
        img2 (N3HW): reconstructed image
        cast_to_int: for evaluating
    Returns:
        three tensor (average psnr values of Y, U, and V)
    """
    if cast_to_int:
        img1 = img1.int()
        img2 = img2.int()
    img1_Y = img1[:, 0:1, :, :]
    img1_U = img1[:, 1:2, ::2, ::2]
    img1_V = img1[:, 2:3, ::2, ::2]
    img2_Y = img2[:, 0:1, :, :]
    img2_U = img2[:, 1:2, ::2, ::2]
    img2_V = img2[:, 2:3, ::2, ::2]
    error_Y = (img1_Y - img2_Y) * (img1_Y - img2_Y)
    error_U = (img1_U - img2_U) * (img1_U - img2_U)
    error_V = (img1_V - img2_V) * (img1_V - img2_V)
    mse_Y_per_image = torch.mean(error_Y.float(), dim=(1, 2, 3))
    psnr_Y_per_image = 10 * torch.log10(255 * 255 / mse_Y_per_image)
    psnr_Y = torch.mean(psnr_Y_per_image)
    mse_U_per_image = torch.mean(error_U.float(), dim=(1, 2, 3))
    psnr_U_per_image = 10 * torch.log10(255 * 255 / mse_U_per_image)
    psnr_U = torch.mean(psnr_U_per_image)
    mse_V_per_image = torch.mean(error_V.float(), dim=(1, 2, 3))
    psnr_V_per_image = 10 * torch.log10(255 * 255 / mse_V_per_image)
    psnr_V = torch.mean(psnr_V_per_image)
    return psnr_Y, psnr_U, psnr_V


class YUVMSE(nn.Module):
    """MSE loss function designed for YUV images.
    """
    def __init__(self):
        super(YUVMSE, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, img1, img2, img_format='rgb'):
        if img_format == 'rgb':
            img1 = torch.clamp((rgb_to_yuv(img1.float() / 255) * 255), 0, 255)
            img2 = torch.clamp((rgb_to_yuv(img2.float() / 255) * 255), 0, 255)
        y1 = img1[:, 0:1, :, :]
        u1 = img1[:, 1:2, :, :]
        v1 = img1[:, 2:3, :, :]
        y2 = img2[:, 0:1, :, :]
        u2 = img2[:, 1:2, :, :]
        v2 = img2[:, 2:3, :, :]
        mse_Y = self.mse(y1, y2)
        mse_U = self.mse(u1, u2)
        mse_V = self.mse(v1, v2)
        return mse_Y, mse_U, mse_V
