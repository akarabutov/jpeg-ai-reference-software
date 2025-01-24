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

import io as IO
import cv2
from typing import Tuple, List, Union

import numpy as np
import torch
from PIL import Image, ImageCms

from .ranges_ops import RangesOps
from .pgx import PGXReader, PGXWriter


class ImageIO:
    @staticmethod
    def read_png(
        file_path: str,
        output_data_range: Tuple[float, float] = [0.0, 1.0],
        bit_depth: int = None,
        device=torch.device('cpu')
    ) -> Tuple[torch.Tensor, str, int]:  # noqa: E125
        """Read PNG file

        Args:
            file_path (str): path to input file
            output_data_range (list, optional): Range of output data. Defaults to [0.0, 1.0].
            bits (int, optional): Bit-depth of input data. Defaults to 8.
            device (torch.device, optional): Device of output . Defaults to torch.device('cpu').

        Returns:
            torch.Tensor: tensor with image
            str: profile
            int: bit-depth of input data
        """

        with Image.open(file_path) as img:
            #rgb_data = np.array(img.convert('RGB'))
            bgr_data = cv2.imread(file_path,  cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
            rgb_data = cv2.cvtColor(bgr_data, cv2.COLOR_BGR2RGB)
            if bit_depth is None:
                bit_depth = 8 if (bgr_data.dtype == np.uint8) else 16

        cur_bd = bit_depth
        if cur_bd is None:
            cur_bd = 8
            bit_depth = 8
        elif cur_bd > 8:
            cur_bd = 16
        max_val = 2**cur_bd - 1
        # Round data by particular bit_depth
        if cur_bd != bit_depth:
            b = cur_bd - bit_depth
            rgb_data >>= b
            rgb_data = rgb_data.round()
            max_val = (2**bit_depth)-1
            
        rgb_tensor_orig_range = torch.tensor(rgb_data.astype(dtype=np.int32), dtype=torch.float,
                                             device=device).permute(2, 0,
                                                                    1).div(max_val).unsqueeze(0)
        rgb_tensor_output_range = RangesOps.denormalize_data(rgb_tensor_orig_range,
                                                             output_data_range)
        return rgb_tensor_output_range.clamp(min(output_data_range),
                                             max(output_data_range)), bit_depth

    @staticmethod
    def write_png(file_path: str,
                  data: torch.Tensor,
                  input_data_range: Tuple[float, float] = [0.0, 1.0],
                  bit_depth: int = 8,
                  bit_shift: int = 0,
                  fill_bit: int = 0) -> None:
        
            data = RangesOps.convert_range(data, input_data_range, (0, (1 << bit_depth) - 1))
            data = data.round()
            data = data.clamp(0, (1 << bit_depth) - 1)
            data *= (1 << bit_shift)
            if fill_bit==1:
                data += (1 << bit_shift)-1
            img = data.squeeze().permute(1, 2, 0).cpu().numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if bit_depth <= 8:
                cv2.imwrite(file_path, img.astype(np.uint8))
            else:
                cv2.imwrite(file_path, img.astype(np.uint16))        
                
    @staticmethod
    def read_pgx(file_path: str) -> List[torch.Tensor]:
        reader = PGXReader(file_path)
        out_tensors = reader.read()        
        return out_tensors
    
    @staticmethod
    def write_pgx(file_path: str,
                  tensors: Union[torch.Tensor, List[torch.Tensor]],
                  float_scale_factor: float = 1.0) -> None:
        writer = PGXWriter(file_path, scale_factor=float_scale_factor)
        writer.write(tensors)