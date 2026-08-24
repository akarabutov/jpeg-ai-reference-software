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
from typing import Dict, Tuple, List

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.codec.common.colorspace import ColorSpace
from src.codec.common.image_io import ImageIO
from src.codec.common.pytorch_ops import TensorOps
from src.codec.common.ranges_ops import RangesOps
from .pgx import PGXWriter

class Image:
    valid_formats: list = ['420', '444', '422']
    valid_subsamplings: dict = {
    #   Format: [ver, hor]
        '444': [1,1],
        '420': [2,2],
        '422': [1,2]
    }
    valid_color_spaces: list = ['rgb', 'yuv']
    valid_comp_names: list = ['a', 'b', 'c']
    shape: list = [0, 0, 0, 0]

    def __init__(self,
                 width: int,
                 height: int,
                 data_range: Tuple[float, float] = [0.0, 1.0],
                 device=torch.device('cpu'),
                 bit_depth=8,
                 format='444',
                 color_space='rgb',
                 data: Dict[str, torch.Tensor] = None):
        """Initialization of a new image

        Args:
            width (int): width of the image
            height (int): height of the image
            profile (dict, optional): Dictionary with image profile. Defaults to {}.
            data_range (list, optional): tuple with
            bit_depth (int, optional): bit-depth of the data. Default is 8.
            format (str, optional): format of data. Default is 444.
            color_space (str, optional): color space of data. Default is rgb.
            data (dict, optional): initial data of tensor
        """
        # name of input file
        self.input_file = None
        # shape of data
        self.shape = [1, 1, height, width]
        # set data range
        self.data_range = data_range
        # set bit-depth
        self.bit_depth = bit_depth
        # validate and set color space
        assert self._validate_color_space(color_space)
        self.color_space = color_space
        # validate and set color space
        assert self._validate_format(format)
        self.format = format
        if data == None:
            self.components = {
                'a': torch.zeros(self.shape[2:], device=device),
                'b': torch.zeros(self.shape[2:], device=device),
                'c': torch.zeros(self.shape[2:], device=device)
            }
        elif isinstance(data, dict):
            self.components = {
                'a': data.get('a', torch.zeros(self.shape[2:], device=device)),
                'b': data.get('b', torch.zeros(self.shape[2:], device=device)),
                'c': data.get('c', torch.zeros(self.shape[2:], device=device))
            }
        elif isinstance(data, torch.Tensor) and data.shape[-3] == 3:
            self.components = {
                'a': data[:,0:1],
                'b': data[:,1:2],
                'c': data[:,2:3]
            }
        else:
            raise NotImplementedError

    def __repr__(self):
        return f"Image: {self.shape} ({self.get_component('a').shape}, {self.get_component('b').shape}, {self.get_component('c').shape}), color space {self.color_space}, format {self.format}, data range {self.data_range}, bit depth {self.bit_depth}"
    
    def get_chroma_subsampling(self) -> Tuple[int]:
        return self.valid_subsamplings[self.format]
    
    @staticmethod
    def get_format_from_subsampling(ver, hor) -> str:
        for k,v in Image.valid_subsamplings.items():
            if ver==v[0] and hor==v[1]:
                return k
        return None
            

    def clone(self) -> 'Image':
        tensor_a = self.get_component('a').clone()
        tensor_b = self.get_component('b').clone()
        tensor_c = self.get_component('c').clone()
        return Image.create_from_tensors(tensor_a,
                                         tensor_b,
                                         tensor_c,
                                         self.data_range,
                                         self.bit_depth,
                                         self.color_space,
                                         format=self.format)

    def zeros_(self) -> None:
        for c in self.components.keys():
            self.components[c].fill_(0.0)

    @property
    def device(self) -> torch.device:
        ans = None
        for d in self.components.values():
            if d is not None:
                ans = d.device
                break
        return ans

    @staticmethod
    def create_from_tensor(tensor: torch.Tensor,
                           data_range: Tuple[float, float] = [0.0, 1.0],
                           bit_depth: int = 8,
                           color_space: str = 'rgb') -> 'Image':
        assert tensor.shape[-3] == 3
        tensor_a = tensor[:, 0:1]
        tensor_b = tensor[:, 1:2]
        tensor_c = tensor[:, 2:3]
        return Image.create_from_tensors(tensor_a, tensor_b, tensor_c, data_range, 
                                         bit_depth, color_space)

    @staticmethod
    def create_from_tensors(tensor_a: torch.Tensor,
                            tensor_b: torch.Tensor,
                            tensor_c: torch.Tensor,
                            data_range: Tuple[float, float] = [0.0, 1.0],
                            bit_depth: int = 8,
                            color_space: str = 'rgb',
                            format: str = '444') -> 'Image':
        valid_t = tensor_a if tensor_a is not None else (
            tensor_b if tensor_b is not None else tensor_c)
        s = valid_t.shape[-2:]
        if tensor_a is None:
            tensor_a = torch.tensor([0])
        if tensor_b is None:
            tensor_b = torch.tensor([0])
        if tensor_c is None:
            tensor_c = torch.tensor([0])
        data = {'a': tensor_a, 'b': tensor_b, 'c': tensor_c}
        ans = Image(s[1],
                    s[0],
                    data_range,
                    tensor_a.device,
                    bit_depth,
                    format=format,
                    color_space=color_space,
                    data=data)
        return ans

    def _validate_format(self, fmt: str) -> bool:
        return fmt in self.valid_formats

    def _validate_color_space(self, clr_spc: str) -> bool:
        return clr_spc in self.valid_color_spaces

    def _validate_comp_name(self, comp_name: str) -> bool:
        return comp_name in self.valid_comp_names

    def check_format(self, fmt: str) -> bool:
        return self.format == fmt

    def check_color_space(self, clr_spc: str) -> bool:
        return self.color_space == clr_spc

    def get_filename(self) -> str:
        return self.input_file

    def get_device(self) -> torch.device:
        return self.components.get('a').device

    def set_data(self,
                 comp_a: torch.Tensor,
                 comp_b: torch.Tensor,
                 comp_c: torch.Tensor,
                 format: str,
                 color_space: str,
                 profile: str = None,
                 bit_depth: int = None) -> None:
        """Set data per component

        Args:
            comp_a (torch.Tensor): data of component a (y in YUV and R in RGB)
            comp_b (torch.Tensor): data of component b (u in YUV and G in RGB)
            comp_c (torch.Tensor): data of component c (v in YUV and B in RGB)
            format (str): input format
            color_space (str): input color space
            profile (str, optional): profile of the data. Defaults to {}.
            bit_depth (int, optional): Bit-depth of the data. Defaults to None.
        """

        assert self._validate_format(format), self._validate_color_space(color_space)
        self.components['a'] = TensorOps.expand_dim_num(comp_a, 4)
        self.components['b'] = TensorOps.expand_dim_num(comp_b, 4)
        self.components['c'] = TensorOps.expand_dim_num(comp_c, 4)
        self.format = format
        self.color_space = color_space
        self.profile = profile
        if bit_depth is not None:
            self.bit_depth = bit_depth

    def _set_data_from_tensor(self, data: torch.Tensor) -> None:
        assert data.shape[-3] == 3
        self.components['a'] = TensorOps.expand_dim_num(data[:, 0:1], 4)
        self.components['b'] = TensorOps.expand_dim_num(data[:, 1:2], 4)
        self.components['c'] = TensorOps.expand_dim_num(data[:, 2:3], 4)

    def _apply_to_components(self, func) -> None:
        for comp in self.components:
            func(self.components[comp])

    def set_component(self, comp_name: str, value: torch.Tensor) -> None:
        assert self._validate_comp_name(comp_name)
        self.components[comp_name] = value

    def get_component(self, comp_name: str) -> torch.Tensor:
        assert self._validate_comp_name(comp_name)
        return self.components.get(comp_name)

    def get_components(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.components.get('a'), self.components.get('b'), self.components.get('c')

    def get_tensor(self) -> torch.Tensor:
        assert self.check_format('444')
        return torch.cat((self.components['a'], self.components['b'], self.components['c']), dim=1)

    def from_tensor_(self,
                     data: torch.Tensor,
                     color_space: str,
                     data_range: Tuple[float, float] = [0.0, 1.0]) -> None:
        assert data.ndim == 4 and data.shape[1] == 3
        self.format = '444'
        self.color_space = color_space
        self.data_range = data_range
        self.components['a'] = data[:, 0:1]
        self.components['b'] = data[:, 1:2]
        self.components['c'] = data[:, 2:3]

    def pad_(self, pad_x, pad_y, mode: str = 'replicate', value: float = 0, comp_list: List[str] = None) -> None:
        if pad_x == 0 and pad_y == 0:
            return
        for c in self.valid_comp_names:
            if comp_list is None or c in comp_list:
                self.components[c] = F.pad(self.components[c], (0, pad_x, 0, pad_y),
                                        mode=mode,
                                        value=value)

    def pad2depth_(self, depth: int) -> None:
        from ..components.base_layers import padding_layer

        for c in self.valid_comp_names:
            v = self.components[c]
            self.components[c] = padding_layer(v, v.shape[-1], v.shape[-2], depth)

    def to_(self, *args, **kwargs) -> None:
        for comp in self.components:
            self.components[comp] = self.components[comp].to(*args, **kwargs)
            
    def is_YUV(self) -> bool:
        return self.check_color_space('yuv')
    
    def is_RGB(self) -> bool:
        return self.check_color_space('rgb')

    def is_420(self) -> bool:
        return self.is_YUV() and self.check_format('420')

    def is_444(self) -> bool:
        return self.is_YUV() and self.check_format('444')

    @staticmethod
    def scale_size(size: torch.Size, scale: float, comp_count:int=2) -> torch.Size:
        ans = list(size[-2:])
        for i in range(comp_count):
            ans[-i - 1] = int(math.ceil(size[-i - 1] * scale))
        return torch.Size(ans)

    @staticmethod
    def calc_chroma_size_444_to_420(size_444: torch.Size) -> torch.Size:
        return Image.scale_size(size_444, 0.5)

    @staticmethod
    def calc_chroma_size_420_to_444(size_420: torch.Size) -> torch.Size:
        return Image.scale_size(size_420, 2.0)
    

    @staticmethod
    def calc_chroma_size_444_to_422(size_444: torch.Size) -> torch.Size:
        return Image.scale_size(size_444, 0.5, 1)

    @staticmethod
    def calc_chroma_size_422_to_444(size_420: torch.Size) -> torch.Size:
        return Image.scale_size(size_420, 2.0, 1)    

    @staticmethod
    def convert_chroma_444_to_420(tensor: torch.Tensor,
                                  mode='bilinear',
                                  new_size=None) -> torch.Tensor:
        if new_size is None:
            new_size = Image.calc_chroma_size_444_to_420(tensor.shape)

        return TensorOps.resize_tensor(tensor, new_size, mode)
    
    def to_420_(self, mode='bilinear') -> None:
        """Convert data to 4:2:0 format
        """
        if self.check_format('444'):
            self.format = '420'
            for c in ['b', 'c']:
                self.components[c] = self.convert_chroma_444_to_420(self.components[c], mode=mode)
        elif self.check_format('420'):
            pass                
        else:
            raise NotImplementedError

    @staticmethod
    def convert_chroma_444_to_422(tensor: torch.Tensor,
                                  mode='bilinear',
                                  new_size=None) -> torch.Tensor:
        if new_size is None:
            new_size = Image.calc_chroma_size_444_to_422(tensor.shape)

        return TensorOps.resize_tensor(tensor, new_size, mode)    


    @staticmethod
    def convert_chroma_420_to_422(tensor: torch.Tensor,
                                  mode='bilinear',
                                  new_size=None) -> torch.Tensor:
        if new_size is None:
            new_size = Image.calc_chroma_size_444_to_422(tensor.shape)

        return TensorOps.resize_tensor(tensor, new_size, mode)    
    
    def to_422_(self, mode='bilinear') -> None:
        """Convert data to 4:2:0 format
        """
        if self.check_format('444'):
            self.format = '422'
            for c in ['b', 'c']:
                self.components[c] = self.convert_chroma_444_to_422(self.components[c], mode=mode)
        elif self.check_format('422'):
            pass
        else:
            raise NotImplementedError


    @staticmethod
    def convert_chroma_420_to_444(tensor: torch.Tensor,
                                  mode='bicubic',
                                  new_size=None) -> torch.Tensor:
        if new_size is None:
            new_size = Image.calc_chroma_size_420_to_444(tensor.shape)

        return TensorOps.resize_tensor(tensor, new_size, mode)
    
    @staticmethod
    def convert_chroma_422_to_444(tensor: torch.Tensor,
                                  mode='bicubic',
                                  new_size=None) -> torch.Tensor:
        if new_size is None:
            new_size = Image.calc_chroma_size_422_to_444(tensor.shape)

        return TensorOps.resize_tensor(tensor, new_size, mode)    

    def to_444_(self, mode='bicubic') -> None:
        """Convert data to 4:4:4 format
        """
        if self.check_format('420'):
            self.format = '444'
            new_height, new_width = self.components['a'].shape[2:]
            for c in ['b', 'c']:
                self.components[c] = Image.convert_chroma_420_to_444(self.components[c],
                                                                     new_size=(new_height,
                                                                               new_width),
                                                                     mode=mode)
        elif self.check_format('422'):
            self.format = '444'
            new_height, new_width = self.components['a'].shape[2:]
            for c in ['b', 'c']:
                self.components[c] = Image.convert_chroma_422_to_444(self.components[c],
                                                                     new_size=(new_height,
                                                                               new_width),
                                                                     mode=mode)            
        elif self.check_format('444'):
            pass
        else:
            raise NotImplementedError
                
    def to_format_(self, format: str) -> None:
        assert format in self.valid_formats
        if format != self.format:
            if format == '420':
                self.to_420_()
            elif format == '444':
                self.to_444_()
            elif format == '422':
                self.to_422_()
            else:
                raise NotImplementedError

    def to_YUV_(self, type='709') -> None:
        if not self.check_color_space('yuv'):
            self.color_space = 'yuv'
            cur_data_range = self.data_range
            self.convert_range_([0.0, 1.0])
            yuv_data = ColorSpace.rgb_to_yuv(self.get_tensor(), type)
            self._set_data_from_tensor(yuv_data)
            self.format = '444'
            self.convert_range_(cur_data_range)

    def to_RGB_(self, type='709') -> None:
        if not self.check_color_space('rgb'):
            cur_data_range = self.data_range
            self.convert_range_([0.0, 1.0])
            self.to_444_()
            rgb_data = ColorSpace.yuv_to_rgb(self.get_tensor(), type)
            self._set_data_from_tensor(rgb_data)
            self.convert_range_(cur_data_range)
            self.color_space = 'rgb'

    def clip_data_(self) -> None:
        """Clip data by data range
        """
        self._apply_to_components(lambda x: x.clamp_(min(self.data_range), max(self.data_range)))

    def round_data_(self) -> None:
        """Round data"""
        self._apply_to_components(lambda x: x.round_())
        
    def round_to_bitdepth_(self, bit_depth: int = None) -> None:
        """Round data to specific bit depth

        Args:
            bit_depth (int, optional): Bit depth. Defaults to the bit depth of the image.
        """
        init_data_range = self.data_range
        if bit_depth is None:
            bit_depth = self.bit_depth
        self.convert_range_([0, (1<<bit_depth)-1])
        self.round_data_()
        self.convert_range_(init_data_range)
        

    def convert_range_(self, new_range: Tuple[float, float]) -> None:
        """Convert data range to 'new_range'

        Args:
            new_range (Tuple[float, float]): new range of internal data
        """
        if (min(new_range) != min(self.data_range)) or \
           (max(new_range) != max(self.data_range)):
            for comp in self.components:
                self.components[comp] = RangesOps.convert_range(self.components[comp],
                                                                self.data_range, new_range)
            self.data_range = new_range

    def write_png(self, file_path: str, bit_depth: int = None, fill_bit:int=0) -> None:
        """Store data to PNG format

        Args:
            file_path (str): path to the output file
            bit_depth (int, optional): output bit-depth of the file. Defaults to None.
            fill_bit (int, optional): value for filling LSB. Defaults to 1.
        """
        if not self.check_color_space("rgb"):
            print("WARNING: the internal format of the data isn't the same as the output image format. Perform convertion of the data to RGB.")
        output_rgb = self.clone()
        output_rgb.to_RGB_()
        output_rgb.clip_data_()
        bit_shift = 0
        if bit_depth is None:
            bit_depth = self.bit_depth
        if bit_depth > 8:
            bit_shift = 16 - bit_depth
        ImageIO.write_png(file_path, output_rgb.get_tensor(), self.data_range, bit_depth, bit_shift, fill_bit)
        
    def write_pgx(self, file_path: str, bit_depth: int = None) -> None:
        scale_val = 2 ** (bit_depth - self.bit_depth) if bit_depth is not None else 1
        output_rgb = self.clone()
        output_rgb.to_RGB_()
        output_rgb.clip_data_()
        ImageIO.write_pgx(file_path, output_rgb.get_tensor(), scale_val)

    def write_file(self, file_path: str, bit_depth: int = None, fill_bit:int=1) -> None:
        """Write image to an output file. Its format is determined from the path.

        Args:
            file_path (str): path to the output file
            bit_depth (int, optional): output bit-depth of the file. Defaults to None.
            fill_bit (int, optional): value for filling LSB. Defaults to 1.

        Raises:
            NotImplementedError: Unknown format of the output file

        """
        if file_path.lower().endswith('.png'):
            self.write_png(file_path, bit_depth, fill_bit)
        elif file_path.lower().endswith(".yuv"):
            self.write_yuv(file_path, bit_depth)
        elif file_path.lower().endswith(".pgx"):
            self.write_pgx(file_path, bit_depth)
        else:
            raise NotImplementedError

    @staticmethod
    def read_png(file_path: str,
                 data_range: Tuple[float, float] = [0.0, 1.0],
                 device=torch.device('cpu'),
                 bit_depth: int = None) -> 'Image':
        """Read PNG file

        Args:
            file_path (str): path to the file
            data_range (Tuple[float, float], optional): expected output data range. Defaults to [0.0, 1.0].
            device (torch.device, optional): device of an output data. Defaults to torch.device('cpu').
            bit_depth (int, optional): expected bit-depth. Defaults to None.
        """
        ext = os.path.splitext(file_path)[1]
        if ext.lower() == '.png':
            data, bit_depth = ImageIO.read_png(file_path,
                                                        output_data_range=data_range,
                                                        bit_depth=bit_depth,
                                                        device=device)
        else:
            raise NotImplementedError
        ans = Image.create_from_tensor(data, data_range, bit_depth, 'rgb')
        ans.input_file = file_path
        return ans

    @staticmethod
    def extract_info(fn, default_bits=10, default_fmt="444"):
        import re

        fn = os.path.basename(fn) # don't macht in directory names
        wh = re.search(r"(?P<w>\d+)x(?P<h>\d+)", fn)
        b = re.search(r"(?P<b>\d+)bit", fn)

        w = wh.group("w") if wh is not None else -1
        h = wh.group("h") if wh is not None else -1

        b = default_bits if b is None else b.group("b")
        fmt = default_fmt
        if "444" in fn:
            fmt = "444"
        elif "420" in fn:
            fmt = "420"
        elif "422" in fn:
            fmt = "422"
        elif "sRGB" in fn:
            fmt = "sRGB"

        return int(w), int(h), int(b), fmt

    @staticmethod
    def read_yuv(
        filename,
        width,
        height,
        bit_depth=8,
        data_range: Tuple[float, float] = [0.0, 1.0],
        fmt="444",
        device="cpu",
    ):
        nr_bytes = int(np.ceil(bit_depth / 8))
        if nr_bytes == 1:
            data_type = np.uint8
        elif nr_bytes == 2:
            data_type = np.uint16
        else:
            raise NotImplementedError(
                "Reading more than 16-bits is currently not supported!"
            )

        yuv_planes = {"Y": None, "U": None, "V": None}
        sizes = {"Y": [height, width], "U": [height, width], "V": [height, width]}

        if fmt == "420":
            for a in ["U", "V"]:
                sizes[a][0] = (sizes[a][0]+1) >> 1
                sizes[a][1] = (sizes[a][1]+1) >> 1
        elif fmt == "400":
            yuv_planes = {"Y": None}
            for a in ["U", "V"]:
                sizes[a][0] = 0
                sizes[a][1] = 0
        elif fmt == "444":
            pass
        elif fmt == "422":
            for a in ["U", "V"]:
                sizes[a][1] = (sizes[a][1]+1) >> 1           
        else:
            raise NotImplementedError("The specified yuv format is not supported!")

        for plane in yuv_planes:
            yuv_planes[plane] = torch.zeros(
                sizes[plane], dtype=torch.float, device=torch.device(device)
            )

        with open(filename, "rb") as f:
            for plane in ["Y", "U", "V"]:
                size = np.int(sizes[plane][0] * sizes[plane][1] * nr_bytes)
                tmp = np.frombuffer(f.read(size), dtype=data_type)
                tmp = tmp.reshape(sizes[plane])
                tmp = tmp[None, None, :]  # expected shape should be 4-dimensional

                yuv_planes[plane] = torch.tensor(
                    (tmp.astype(np.float32) / (2**bit_depth - 1))
                    * (max(data_range) - min(data_range))
                    + min(data_range),
                    dtype=torch.float,
                    device=torch.device(device),
                )

        image = Image.create_from_tensors(
            yuv_planes["Y"],
            yuv_planes["U"],
            yuv_planes["V"],
            bit_depth=bit_depth,
            color_space="yuv",
            format=fmt,
        )
        image.input_file = filename

        return image

    def write_yuv(self, file_path: str, bit_depth: int = None) -> None:
        # assert self.check_format('420')
        if not self.check_color_space("yuv"):
            print("WARNING: the internal format of the data isn't the same as the output image format. Perform convertion of the data to YUV.")

        if bit_depth is None:
            bit_depth = self.bit_depth
            
        _, _, bit_depth, fmt = self.extract_info(file_path, bit_depth)
        # yuv = self.yuv_data.copy()
        nr_bytes = np.ceil(bit_depth / 8)
        if nr_bytes == 1:
            data_type = np.uint8
        elif nr_bytes == 2:
            data_type = np.uint16
        elif nr_bytes <= 4:
            data_type = np.uint32
        else:
            raise NotImplementedError(
                "Writing more than 16-bits is currently not supported!"
            )

        # rescale to range of bits
        output_yuv = self.clone()
        output_yuv.to_YUV_()
        output_yuv.to_format_(fmt)
        output_yuv.convert_range_((0, (1 << bit_depth) - 1))
        output_yuv.round_data_()
        output_yuv.clip_data_()
        # dump to file
        lst = []
        for plane in output_yuv.get_components():
            # if plane in yuv.keys():
            plane = plane.cpu().numpy()
            lst = lst + plane.ravel().tolist()

        raw = np.array(lst)

        raw.astype(data_type).tofile(file_path)

    @staticmethod
    def read_file(
        file_path: str,
        data_range: Tuple[float, float] = [0.0, 1.0],
        device=torch.device("cpu"),
        bit_depth: int = None,
        default_yuv_fmt: str = '420'
    ) -> "Image":
        """Read image from a file. Its format is determined from the path.

        Args:
            file_path (str): path to the file
            data_range (Tuple[float, float], optional): expected output data range. Defaults to [0.0, 1.0].
            device (torch.device, optional): device of an output data. Defaults to torch.device('cpu').
            bit_depth (int, optional): expected bit-depth. Defaults to 8.
            default_yuv_fmt (str, optional): default YUV format if the format of YUV cannot be extracted from filename

        Raises:
            NotImplementedError: unknown file format

        Returns:
            Image: object with image
        """
        
        if file_path.lower().endswith('.png'):
            return Image.read_png(file_path, data_range, device, bit_depth)
        elif file_path.lower().endswith("yuv"):
            w, h, b, fmt = Image.extract_info(file_path, default_bits=8, default_fmt=default_yuv_fmt)
            return Image.read_yuv(file_path, w, h, b, data_range, fmt, device)
        else:
            raise NotImplementedError

    def get_hashs(self) -> Tuple[str]:
        """Get tuple with MD5 hashs of image components

        Returns:
            Tuple[str]: MD5('a'), MD5('b'), MD5('c')
        """
        ans = list()
        for c in self.get_components():
            ans.append(TensorOps.get_hash(c))
        return ans
