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
from typing import List

import torch

from src.codec.common import Image
from src.codec.entropy_coding import HeaderCoder

from ..interfaces import Transformer
from ..resampler import Resampler
##
from .params import RCParams


class ResolutionChanger(Transformer):
    def __init__(self, *args, **kwargs):
        super(ResolutionChanger, self).__init__(has_enabled_flag=True, *args, **kwargs)
        self.resampler = Resampler()
        self._params_rc = RCParams()

        self.intra_coding_width_chroma = -1
        self.intra_coding_height_chroma = -1
        self.img_orig_width = -1
        self.img_orig_height = -1
        self.is_encoder = True
        if 'is_encoder' in kwargs:
            self.is_encoder = kwargs['is_encoder']

    #def init_decoder(self):
    #    self.res_changer.setup([self.img_height, self.img_width])

    def setup(self, img_shape: List[int]):
        self.img_orig_height = img_shape[0]
        self.img_orig_width = img_shape[1]
        self.intra_coding_width = self.img_orig_width
        self.intra_coding_height = self.img_orig_height
        if self.is_enabled():
            assert not((self.scale_hor == 1) and (self.scale_ver == 2))
            self.intra_coding_width_chroma = (self.img_orig_width +self.scale_hor-1)>>(self.scale_hor-1)
            self.intra_coding_height_chroma = (self.img_orig_height +self.scale_ver-1)>>(self.scale_ver-1)
            # make sure we do not make size larger than original due to alignment
        else:
            self.scale_hor = 2
            self.scale_ver = 2
            self.intra_coding_height_chroma = (self.img_orig_height +self.scale_ver-1)>>(self.scale_ver-1)
            self.intra_coding_width_chroma =  (self.img_orig_width +self.scale_hor-1)>>(self.scale_hor-1)

    def encode_header(self, ec: HeaderCoder):
        ec.encode(self.scale_ver-1, max_symbol_value=1, name='scale_ver')
        ec.encode(self.scale_hor-1, max_symbol_value=1, name='scale_hor')
        self.logger.debug(
            'Intra coding resolution change enabled. Original img shape: %s. Intra coding img shape: %s'
            % (self.get_original_img_shape(), self.get_processed_img_shape()))

    def decode_header(self, ec: HeaderCoder):
        self.scale_ver = int(
            ec.decode([1], max_symbol_value=1, name='scale_ver'))+1
        self.scale_hor = int(
            ec.decode([1], max_symbol_value=1, name='scale_hor'))+1
        self.logger.debug(
            'Intra coding resolution change enabled. Original img shape: %s. Intra coding img shape: %s'
            % (self.get_original_img_shape(), self.get_processed_img_shape()))

    def get_processed_img_shape(self):
        """Return img shape for intra coding.

        Returns:
            torch.Size: Shape of an intra img of height h and width w as (h, w).
        """
        return torch.Size([self.intra_coding_height, self.intra_coding_width])

    def get_original_img_shape(self):
        return torch.Size([self.img_orig_height, self.img_orig_width])

    def _forward_transform(self, img: Image) -> Image:
        img_shape = self.get_processed_img_shape()
        if img.shape[-2:] == list(img_shape):
            return img

        resampled_img = Image(width=img_shape[-1],
                              height=img_shape[-2],
                              data_range=img.data_range,
                              profile=img.profile,
                              device=img.device,
                              bit_depth=img.bit_depth,
                              format=img.format,
                              color_space=img.color_space)

        h_size_aligned = img_shape[-2]
        w_size_aligned = img_shape[-1]

        for c in Image.valid_comp_names:
            resampled_img.set_component(
                c,
                self.resampler.resize_luma(img.get_component(c), h_size_aligned, w_size_aligned,
                                           True))
        resampled_img.clip_data_()

        return resampled_img

    def _backward_transform(self, img) -> Image:
        img_shape = self.get_original_img_shape()
        if self.scale_ver == 1 and self.scale_hor == 1:
            return img
        if self.scale_ver == 2 and img.get_component('a').shape[-2] == img.get_component('b').shape[-2]:
            return img #upsampling already done.
        if self.scale_hor == 2 and img.get_component('a').shape[-1] == img.get_component('b').shape[-1]:
            return img #upsampling already done.
        old_data_range = img.data_range
        img.convert_range_([0, 1])
        tensor_shape = img.shape[:2] + list(img_shape[-2:])
        resampled_img = torch.zeros(tensor_shape, device=img.device)

        h_size_orig = img_shape[-2]
        w_size_orig = img_shape[-1]
        h_size_orig_luma = h_size_orig
        w_size_orig_luma = w_size_orig
        resampled_img = Image(width=w_size_orig,
                              height=h_size_orig,
                              data_range=img.data_range,
                              profile=img.profile,
                              device=img.device,
                              bit_depth=img.bit_depth,
                              format=img.format,
                              color_space=img.color_space)

        for c in Image.valid_comp_names:
            if not(c == 'a'):
                h_size = self.intra_coding_height_chroma*self.scale_hor
                w_size = self.intra_coding_width_chroma*self.scale_ver
            else:
                h_size = h_size_orig_luma
                w_size = w_size_orig_luma
            resampled_img.set_component(
                c, self.resampler.resize_luma(img.get_component(c), h_size, w_size,
                                              False)[:,:,:h_size_orig_luma,:w_size_orig_luma])
            if  resampled_img.get_component('a').shape[-1] == resampled_img.get_component('b').shape[-1]:
                 resampled_img.format = '444'
            elif resampled_img.get_component('a').shape[-2] == resampled_img.get_component('b').shape[-2]:
                resampled_img.format = '422'
            else:
                resampled_img.format = '420'

        resampled_img.clip_data_()
        resampled_img.convert_range_(old_data_range)
        return resampled_img
