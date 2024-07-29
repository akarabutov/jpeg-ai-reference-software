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

from .conv_layers import (MaskedConv2d, LightResidualBlock, ResidualBlock, ResidualBlock_BN,
                          ResidualBlock_BN_RectKernel, conv1x1, conv1x1i, conv3x3, conv3x3i, conv3x3_t, conv4x4_t, LightCombineBlock, conv2x2_pxl2)
from .conv_quant_layers import Conv2di
from .quant_layer import QuantModule
from .rnab import RNAB, SlimmedRNAB
from .cab import CAB
from .tam import TAM
from .utils import (clip_image, clip_image_sgl, cropping_layer, denormalize, make_layer, normalize,
                    padding_layer, feature_clipping, get_size_on_depth)

__all__ = [
    'conv1x1',
    'conv1x1i',
    'conv3x3',
    'conv3x3i',
    'conv3x3_t',
    'conv4x4_t',
    'conv2x2_pxl2',
    'get_size_on_depth',
    'MaskedConv2d',
    'ResidualBlock',
    'LightResidualBlock',
    'RNAB',
    'SlimmedRNAB',
    'ResidualBlock_BN',
    'ResidualBlock_BN_RectKernel',
    'QuantModule',
    'Conv2di',
    'clip_image',
    'clip_image_sgl',
    'cropping_layer',
    'padding_layer',
    'feature_clipping', 
    'denormalize',
    'normalize',
    'make_layer',
    'CAB',
    'TAM',
    'LightCombineBlock'
]
