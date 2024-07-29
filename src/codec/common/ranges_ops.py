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

from typing import Tuple

import torch


class RangesOps:
    """Basic operations with data ranges
    """
    @staticmethod
    def calculate_full_range(range: Tuple[float, float]) -> float:
        return max(range) - min(range)

    @staticmethod
    def normalize_data(data: torch.Tensor, input_range: Tuple[float, float]) -> torch.Tensor:
        """Normalize input range of data, i.e. convert it to range [0,1]

        Args:
            data (torch.Tensor): input data
            input_range (Tuple[float, float]): range of input data

        Returns:
            torch.Tensor: normalized data
        """
        input_full_range = RangesOps.calculate_full_range(input_range)
        return (data - min(input_range)) / input_full_range

    @staticmethod
    def denormalize_data(normalized_data: torch.Tensor,
                         output_range: Tuple[float, float]) -> torch.Tensor:
        """Convert normalized data (in range [0,1]) to specific output range

        Args:
            normalized_data (torch.Tensor): normalized input data
            output_range (Tuple[float, float]): desirable output data range

        Returns:
            torch.Tensor: output denormalized data
        """
        output_full_range = RangesOps.calculate_full_range(output_range)
        return normalized_data * output_full_range + min(output_range)

    @staticmethod
    def convert_range(data: torch.Tensor, input_range: Tuple[float, float],
                      output_range: Tuple[float, float]) -> torch.Tensor:
        """Convert data from one range to another one

        Args:
            data (torch.Tensor): input data
            input_range (Tuple[float, float]): range of input data
            output_range (Tuple[float, float]): desirable output data range

        Returns:
            torch.Tensor: output data
        """
        normalized_data = RangesOps.normalize_data(data, input_range)
        return RangesOps.denormalize_data(normalized_data, output_range)
