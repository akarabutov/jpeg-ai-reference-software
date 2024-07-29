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

from ..interfaces import BaseEngine
from ..interpolation import create_interpolation_instance
from ..interpolation.ReconResample import BiCubic, Lanczos
##
from .params import ResamplerParams


class Resampler(BaseEngine):
    def __init__(self, **kwargs):
        super(Resampler, self).__init__(has_enabled_flag=False, **kwargs)
        self._resampler_params = ResamplerParams()
        self.interpolator_cubic = BiCubic()
        self.interpolator_lanc = Lanczos()
        ''' #SEMIH: according to our understanding create_interpolation_instance() implementation is wrong. ToolsComposite does not have implement function. and the interpolators
        #do not implement an is_enabled() method.
        self.interpolator_luma_up = create_interpolation_instance()
        self.interpolator_luma_dwn = create_interpolation_instance()
        self.interpolator_chroma_up = create_interpolation_instance()
        self.interpolator_chroma_dwn = create_interpolation_instance()
        self.interpolator_luma_scale_factor_close_to_one = create_interpolation_instance()
        self.interpolator_chroma_scale_factor_close_to_one = create_interpolation_instance()
        '''

    def _decide_interpolator(self, isEncoder):
        if isEncoder:
            return self.interpolator_lanc
        if self.fastResize:
            return self.interpolator_cubic
        return self.interpolator_lanc

    def _resize(self, input, H_out, W_out, isEncoder):

        B, C, H_in, W_in = input.shape
        new_size = [B, C, H_out, W_out]
        if H_in == H_out and W_in == W_out:
            return input.clone()

        interpolator = self._decide_interpolator(isEncoder)

        output = interpolator(input, new_size)

        return output

    def resize_luma(self, input, H_out, W_out, isEncoder=True):
        return self._resize(input, H_out, W_out, isEncoder=isEncoder)

    def resize_chroma(self, input, H_out, W_out, isEncoder=True):
        return self._resize(input, H_out, W_out, isEncoder=isEncoder)

    def resize_img(self, input, H_L_out, W_L_out, H_C_out, W_C_out):

        out_img = {}

        out_img['Y'] = self._resize(input['Y'],
                                    H_L_out,
                                    W_L_out,
                                    is_luma=True,
                                    align_corners=self.align_corners_luma)

        UV_in = torch.cat((input['U'], input['V']), dim=1)

        UV_out = self._resize(UV_in,
                              H_C_out,
                              W_C_out,
                              is_luma=False,
                              align_corners=self.align_corners_chroma)

        out_img['U'] = UV_out[:, 0:1]
        out_img['V'] = UV_out[:, 1:2]

        return out_img

    def _resize_pil(self, input, H_out, W_out):
        import PIL
        import torchvision.transforms.functional as TF

        B, C, H, W = input.shape

        device = input.device

        output = torch.zeros(B, C, H_out, W_out, device=device)

        for b in range(B):
            for c in range(C):
                xYPil = TF.to_pil_image(input[b, c].detach().cpu(), mode='F')
                xYPil = TF.resize(xYPil, (H_out, W_out), interpolation=PIL.Image.LANCZOS)
                output[b, c] = TF.to_tensor(xYPil).to(device)

        return output
