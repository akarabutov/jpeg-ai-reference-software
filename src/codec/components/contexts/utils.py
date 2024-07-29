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
import numpy
from typing import Tuple


class ContextUtils:

    @staticmethod
    def channel_wise_pad(data: torch.Tensor, C2: int) -> torch.Tensor:
        """Channel-wise padding of the input tensor up to C2 channels

        Args:
            data (torch.Tensor): the input tensor
            C2 (int): new number of channels

        Returns:
            torch.Tensor: output tensor
        """
        C1 = data.shape[-3]
        pad_C = C2 - C1
        new_shape = list(data.shape)
        new_shape[-3] = pad_C
        zeros = torch.zeros(new_shape, device=data.device, dtype=data.dtype)
        return torch.cat( (data, zeros), dim=1 )

    @staticmethod
    def down_shuffle(y: torch.Tensor, factor_hw: int = 2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Downsample and shuffle of the input tensor (y) on 4 parts. See figure E.2 in WD

        Args:
            y (torch.Tensor): input tensor
            factor_hw (int, optional): factor of downsampling. Defaults to 2.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: slices of input tensor
        """
        B, iC, iH, iW = y.shape
        oC, oH, oW = iC * (factor_hw * factor_hw), iH // factor_hw, iW // factor_hw

        y = y.reshape(B, iC, oH, factor_hw, oW, factor_hw)
        y = y.permute(0, 1, 2, 4, 3, 5)  # B, iC, pH, pW, oH, oW
        y = y.reshape(B, iC, oH, oW, factor_hw * factor_hw)
        part1, part2, part3, part4 = torch.chunk(y, chunks=4, dim=4)
        return part1.squeeze(4), part4.squeeze(4), part2.squeeze(4), part3.squeeze(4)

    @staticmethod
    def up_shuffle(input_rec: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], factor_hw: int = 2) -> torch.Tensor:
        """Upsampling and unshuffle of the input tensors. See figure E.2 in WD

        Args:
            input_rec (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): input tensors
            factor_hw (int, optional): factor of upsampling. Defaults to 2.

        Returns:
            torch.Tensor: unshuffled and upsampled tensor
        """
        part0, part3, part1, part2 = input_rec
        b, c, h, w = part0.shape  ##
        data = torch.stack((part0, part1, part2, part3), dim=4).reshape(b, c, h, w, factor_hw, factor_hw)
        data = data.permute(0, 1, 2, 4, 3, 5).reshape(b, c, factor_hw * h, factor_hw * w)
        return data

def Upsample_proc(inputs):
    ret = ContextUtils.up_shuffle(inputs)
    return ret

class HyperToContext9x1b(torch.nn.Module):
    def __init__(self):
        super(HyperToContext9x1b, self).__init__()

    def forward(self, x):
        slice_1, slice_2, slice_3, slice_4 = torch.chunk(x, 4, dim=1)
        return slice_1, slice_2, slice_3, slice_4
