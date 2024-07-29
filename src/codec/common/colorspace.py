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

import numpy as np
import torch
from torch.nn import functional

__all__ = ['ColorSpace']


class YuvStd:
    params_dict = {
        '601': {
            'Kr': 0.2990,
            'Kg': 0.5870,
            'Kb': 0.1140,
            'Kby': 1.7720,
            'Kry': 1.4020
        },
        '709': {
            'Kr': 0.2126,
            'Kg': 0.7152,
            'Kb': 0.0722,
            'Kby': 1.8556,
            'Kry': 1.5748
        },
        '2020': {
            'Kr': 0.2627,
            'Kg': 0.6780,
            'Kb': 0.0593,
            'Kby': 1.8814,
            'Kry': 1.4747
        },
    }

    delta = 0.5

    @staticmethod
    def rgb2yuv(rgb, yuv_type='709'):
        """rgb2yuv

        Args:
            rgb: shape=[..., C, *, *]
            yuv_type:

        Returns:
            yuv_list: a list of channel instances.
        """
        r = rgb[..., 0, :, :]
        g = rgb[..., 1, :, :]
        b = rgb[..., 2, :, :]

        params = YuvStd.get_params(yuv_type)
        y = params['Kr'] * r + params['Kg'] * g + params['Kb'] * b
        u = (b - y) / params['Kby'] + YuvStd.delta
        v = (r - y) / params['Kry'] + YuvStd.delta

        yuv_list = [y, u, v]
        return yuv_list

    @staticmethod
    def yuv2rgb(yuv, yuv_type='709'):
        """yuv2rgb

        Args:
            yuv: shape=[..., C, *, *]
            yuv_type:

        Returns:
            rgb_list: a list of channel instances.
        """
        y = yuv[..., 0, :, :]
        u = yuv[..., 1, :, :] - YuvStd.delta
        v = yuv[..., 2, :, :] - YuvStd.delta

        params = YuvStd.get_params(yuv_type)
        r = y + (params['Kry'] * v)
        g = y - (params['Kb'] * params['Kby'] / params['Kg']) * u - (params['Kr'] * params['Kry'] /
                                                                     params['Kg']) * v
        b = y + (params['Kby'] * u)

        rgb_list = [r, g, b]
        return rgb_list

    @staticmethod
    def get_params(yuv_type):
        return YuvStd.params_dict[yuv_type]


class ColorSpace:
    """ColorSpace for numpy
    """
    @staticmethod
    def rgb_to_yuv(rgb, yuv_type: str = '709'):
        """Convert an image from RGB to YUV.
        The image data is assumed to be in the range of (0, 1).

        Args:
            rgb: shape (*, 3, H, W), data_types=(np.ndarray, torch.Tensor).
            yuv_type: values=YuvStd.params_dict.keys.
        Returns:
            yuv: YUV Image with shape (*, 3, H, W).
        """
        data_type = ColorSpace.check_data_type(rgb)
        ColorSpace.check_yuv_type(yuv_type)
        yuv_list = YuvStd.rgb2yuv(rgb, yuv_type)
        yuv = ColorSpace.fuse_channels(data_type, yuv_list)
        return yuv

    @staticmethod
    def yuv_to_rgb(yuv, yuv_type: str = '709'):
        """ Convert an image YUV to RGB.
        The image data is assumed to be in the range of (0, 1).

        Args:
            yuv: shape (*, 3, H, W), data_types=(np.ndarray, torch.Tensor).
            yuv_type: values=YuvStd.params_dict.keys.
        Returns:
            rgb: RGB Image with shape (*, 3, H, W).
        """
        data_type = ColorSpace.check_data_type(yuv)
        ColorSpace.check_yuv_type(yuv_type)
        rgb_list = YuvStd.yuv2rgb(yuv, yuv_type)
        rgb = ColorSpace.fuse_channels(data_type, rgb_list)
        return rgb

    @staticmethod
    def fuse_channels(data_type: str, ch_list: list):
        """fuse_channels

        Args:
            data_type: ['np.ndarray', 'torch.Tensor']
            ch_list: a list of channel instances.

        Returns:
            chs: a instance of fused channels.
        """
        if data_type == 'np.ndarray':
            chs = np.concatenate(ch_list, axis=-3)
        elif data_type == 'torch.Tensor':
            chs = torch.stack(ch_list, dim=-3)
        else:
            raise AssertionError('Invalid data_type={}'.format(data_type))

        return chs

    @staticmethod
    def check_data_type(data):
        if isinstance(data, np.ndarray):
            if (data.ndim < 3) or (data.shape[-3] != 3):
                raise ValueError(
                    'Input size must have shape=(*, 3, H, W) but with shape={}'.format(data.shape))
            data_type = 'np.ndarray'

        elif isinstance(data, torch.Tensor):
            if (len(data.shape) < 3) or (data.shape[-3] != 3):
                raise ValueError(
                    'Input size must have shape=(*, 3, H, W) but with shape={}'.format(data.shape))
            data_type = 'torch.Tensor'

        else:
            raise TypeError(
                'Data type must be np.ndarray/torch.Tensor but with data_type={}'.format(
                    type(data)))

        return data_type

    @staticmethod
    def check_yuv_type(yuv_type):
        if yuv_type not in YuvStd.params_dict:
            raise AssertionError(
                'YUV type must be supported but with yuv_type={}'.format(yuv_type))


def yuv_420_to_444(y, u, v):
    def _upsample_nearest_neighbor(x, factor=2):
        y = functional.interpolate(x.unsqueeze(0),
                                   scale_factor=factor,
                                   mode='bilinear',
                                   align_corners=True)
        y = y.squeeze(0)
        return y

    u, v = map(_upsample_nearest_neighbor, (u, v))  # upsample U, V
    yuv = torch.cat((y, u, v), dim=0)  # merge Y, U, V
    return yuv
