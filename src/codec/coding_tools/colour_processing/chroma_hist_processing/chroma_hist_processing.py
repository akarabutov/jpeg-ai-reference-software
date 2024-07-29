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
from .params import ChromaHistProcessingParams
##
from ..base import ColorProcessingBase


class ChromaHistProcessing(ColorProcessingBase):
    def __init__(self, **kwargs):
        super(ChromaHistProcessing, self).__init__(use_coding_headers=False, **kwargs)
        self._params_chp = ChromaHistProcessingParams()

    def pre_processing(self, img: Image, *args, **kwargs) -> Image:
        mean1 = torch.mean((img.get_component('b')))
        mean2 = torch.mean((img.get_component('c')))
        chroma_u = self.coeff*(img.get_component('b')-mean1)+mean1
        chroma_v = self.coeff*(img.get_component('c')-mean2)+mean2
        return Image.create_from_tensors(img.get_component('a'),
                                         chroma_u,
                                         chroma_v, 
                                         data_range=img.data_range,
                                         bit_depth=img.bit_depth,
                                         color_space=img.color_space,
                                         format=img.format)
        

