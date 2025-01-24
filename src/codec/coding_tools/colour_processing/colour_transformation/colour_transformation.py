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

from src.codec.common import Image
from src.codec.entropy_coding import ECModule, HeaderCoder
from src.codec.common.utils import pop_param_from_dict

##
from ..base import ColorProcessingBase
from .params import ColorTransformParams


class ColourTransformation(ColorProcessingBase):
    def __init__(self, **kwargs):
        kwargs, _ = pop_param_from_dict(kwargs, 'has_enabled_flag', False)
        super(ColourTransformation, self).__init__(has_enabled_flag=False, **kwargs)
        self._params_color = ColorTransformParams()
        self.inv_matrix = torch.eye(3)
        
    def _params_loaded(self) -> None:
        assert (min(self.colour_transform_offset) >= 0) and (max(self.colour_transform_offset) <= 255)
        assert (min(self.colour_transform_matrix) >= 0) and (max(self.colour_transform_matrix) <= 255)
        self.clr_tr_offset = torch.tensor(self.colour_transform_offset, dtype=torch.int)
        self.clr_tr_matrix = torch.tensor(self.colour_transform_matrix, dtype=torch.int).view((3,3))
        self.update_inverse_matrix()
        
    def update_inverse_matrix(self) -> None:
        self.inv_matrix = torch.inverse(self.clr_tr_matrix / 255.0) * 255.0

    def pre_processing(self, img: Image, *args, **kwargs) -> Image:
        ans = img
        if self.colour_transform_idx is None:
            self.colour_transform_idx = 0 if img.is_YUV() else 1
                
        if self.colour_transform_idx == 1:
            ans = img.clone()
            ans.to_YUV_()
        elif self.colour_transform_idx == 2:
            self.inv_matrix = self.inv_matrix.to(img.device)
            self.clr_tr_offset = self.clr_tr_offset.to(img.device)
            
            s_ver = self.get_owner_param('s_ver')
            s_hor = self.get_owner_param('s_hor')
            
            ans.convert_range_(0,1)
            ia = ans.get_component('a')
            ib = ans.get_component('b')
            ic = ans.get_component('c')
            
            na = (ia * self.inv_matrix[0,0] + ib * self.inv_matrix[0,1] + ic * self.inv_matrix[0,2] - self.clr_tr_offset[0]) / 255.0
            nb = (ia[..., ::s_ver,::s_hor] * self.inv_matrix[0,0] + ib * self.inv_matrix[0,1] + ic * self.inv_matrix[0,2] - self.clr_tr_offset[1]) / 255.0
            nc = (ia[..., ::s_ver,::s_hor] * self.inv_matrix[0,0] + ib * self.inv_matrix[0,1] + ic * self.inv_matrix[0,2] - self.clr_tr_offset[2]) / 255.0
            na.clamp_(0,1)
            nb.clamp_(0,1)
            nc.clamp_(0,1)
            
            ans.set_component('a', na)
            ans.set_component('b', nb)
            ans.set_component('c', nc)

        return ans



    def post_processing(self, img: Image, *args, **kwargs) -> Image:
        ans = img.clone()
        if self.colour_transform_idx == 0:
            pass 
        elif self.colour_transform_idx == 1:
            ans.to_RGB_()
        elif self.colour_transform_idx == 2:

            s_ver = self.get_owner_param('s_ver')
            s_hor = self.get_owner_param('s_hor')
            bd = self.get_owner_param('image_data_bits')
            self.clr_tr_matrix = self.clr_tr_matrix.to(img.device)            
            self.clr_tr_offset = self.clr_tr_offset.to(img.device)


            s_ver = self.get_owner_param('s_ver')
            s_hor = self.get_owner_param('s_hor')
            
            ans.convert_range_(0,1)
            ia = ans.get_component('a')
            ib = ans.get_component('b')
            ic = ans.get_component('c')
            
            na = (ia * self.inv_matrix[0,0] + ib * self.inv_matrix[0,1] + ic * self.inv_matrix[0,2] + self.clr_tr_offset[0]) / 255.0
            nb = (ia[..., ::s_ver,::s_hor] * self.inv_matrix[0,0] + ib * self.inv_matrix[0,1] + ic * self.inv_matrix[0,2] + self.clr_tr_offset[1]) / 255.0
            nc = (ia[..., ::s_ver,::s_hor] * self.inv_matrix[0,0] + ib * self.inv_matrix[0,1] + ic * self.inv_matrix[0,2] + self.clr_tr_offset[2]) / 255.0
            
            ans.set_component('a', na)
            ans.set_component('b', nb)
            ans.set_component('c', nc)
            ans.round_to_bitdepth_(bd)
        ans.clip_data_()

        return ans

    def encode_header(self, ec: HeaderCoder):
        ec.encode(self.colour_transform_idx, 2, name='colour_transform_idx')
        if self.colour_transform_idx == 2:
            for i in range(3):
                for j in range(3):
                    ec.encode(self.colour_transform_matrix[3*i+j], bits_count=8, name=f'colour_transform_matrix[{i}][{j}]')
            for i in range(3):
                ec.encode(self.colour_transform_offset[i], bits_count=8, name=f'colour_transform_offset[{i}]')

    def decode_header(self, ec: HeaderCoder):
        self.colour_transform_idx = ec.decode([1], 2, name='colour_transform_idx').item()
        if self.colour_transform_idx == 2:
            for i in range(3):
                for j in range(3):
                    self.colour_transform_matrix[3*i+j] = (ec.decode([1], bits_count=8, name=f'colour_transform_matrix[{i}][{j}]').item())
            for i in range(3):
                self.colour_transform_offset[i] = (ec.decode([1], bits_count=8, name=f'colour_transform_offset[{i}]').item())
            self._params_loaded()
            self.update_inverse_matrix()
