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
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

##
from .base.interface import InterpolationFilterBase


class BiCubic(InterpolationFilterBase):
    def forward(self, x, size):
        assert isinstance(x, torch.Tensor) and len(x.size()) == 4
        h, w = size[-2:]
        wx, hx = x.size(3), x.size(2)
        if w == wx and h == hx:
            return x
        res = transforms.Resize(size=(h, w), interpolation=transforms.InterpolationMode.BICUBIC)
        tensor_resized = res(x)
        return tensor_resized


class Lanczos(InterpolationFilterBase):
    def forward(self, x, size):
        assert isinstance(x, torch.Tensor) and len(x.size()) == 4
        h, w = size[-2:]
        wx, hx = x.size(3), x.size(2)
        if w == wx and h == hx:
            return x
        img = TF.to_pil_image(x.squeeze().detach().cpu(), mode='F')
        img_resized = img.resize(size=(w, h), resample=Image.LANCZOS)
        tensor_resized = TF.to_tensor(img_resized).unsqueeze(0).to(x.device)
        return tensor_resized
