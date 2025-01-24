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
from functools import wraps
from typing import Callable, Union, List
from ..base import FilterBase
from src.codec.common import Image, Decisions
from src.codec.entropy_coding import ECModule
from .params import LEFParams


class LEF(FilterBase):
    """
        Implements the filter bank interface containing LEF filters.
    """

    def __init__(self, **kwargs):
        super(LEF, self).__init__(enable_flag_name="LEF_enabled_flag", **kwargs)
        self._params_ = LEFParams()
        self.recSharpMagList = [[1.10, 1.13, 1.16],
                                [1.07, 1.09, 1.11],
                                [1.03, 1.04, 1.05],
                                [1.01, 1.02, 1.03]]
        self.recSharpThrList = [[600, 1200, 2000],
                                [800, 1400, 2400],
                                [1000, 1600, 2800],
                                [1200, 2000, 3200]]

    def get_decision_y(self, decisions: Decisions) -> Decisions:
        return decisions.get(self.get_base_model_name(), dict()).get('model_y', dict)

    def analyze(self, decisions: Decisions) -> int:
        scale_log = decisions.get('scale_log', None)
        avg_sig = torch.mean(scale_log.float(), dim=[2, 3])
        reference_channel_idx = int(torch.argmax(avg_sig).item())
        return reference_channel_idx

    def decode_header(self, ec: ECModule):
        self.reference_channel_idx = -1
        self.reference_channel_idx = int(ec.decode([1], max_symbol_value=255, name='LEF_chIdx'))

    def encode_header(self, ec: ECModule):
        ec.encode(int(self.reference_channel_idx), max_symbol_value=255, name='LEF_chIdx')

    def compress(self, rec_imgs: List[Image], org_img_i: Image, *args, **kwargs) -> List[Image]:
        rec = rec_imgs[0]
        decisions_Y = self.get_decision_y(kwargs.get('decisions', dict()))

        reference_channel_idx = self.analyze(decisions_Y)
        self.reference_channel_idx = reference_channel_idx
        model_id = self.get_base_model_id()

        thr1 = self.recSharpThrList[model_id][0]
        thr2 = self.recSharpThrList[model_id][1]
        thr3 = self.recSharpThrList[model_id][2]
        mag1 = self.recSharpMagList[model_id][0]
        mag2 = self.recSharpMagList[model_id][1]
        mag3 = self.recSharpMagList[model_id][2]

        rec_Y = rec.get_component('a')
        rec_U = rec.get_component('b')
        rec_V = rec.get_component('c')
        rec_Y = rec_Y / 255.0

        scale_hat = decisions_Y.get('scale_log')[:, reference_channel_idx:reference_channel_idx + 1, :, :].clone()
        scale_hat = scale_hat.to(dtype=torch.float32)
        (_, _, h, w) = rec_Y.shape
        s_upsampled = torch.nn.functional.interpolate(scale_hat, size=(h, w), mode='nearest')
        mag_list = [mag1, mag2, mag3]
        thr_list = [thr1, thr2, thr3]
        rec_Y = adptive_sharpness(rec_Y, s_upsampled, mag_list, thr_list)
        rec_Y = rec_Y * 255.0

        ans = Image.create_from_tensors(rec_Y,
                                        rec_U,
                                        rec_V,
                                        rec.data_range,
                                        bit_depth=rec.bit_depth,
                                        format=rec.format,
                                        color_space='yuv')

        return [ans] + rec_imgs[1:]

    def decompress(self, imgs: List[Image], *args, **kwargs) -> List[Image]:
        img = imgs[0]
        reference_channel_idx = self.reference_channel_idx
        model_id = self.get_base_model_id()

        thr1 = self.recSharpThrList[model_id][0]
        thr2 = self.recSharpThrList[model_id][1]
        thr3 = self.recSharpThrList[model_id][2]
        mag1 = self.recSharpMagList[model_id][0]
        mag2 = self.recSharpMagList[model_id][1]
        mag3 = self.recSharpMagList[model_id][2]

        rec_Y = img.get_component('a')
        rec_U = img.get_component('b')
        rec_V = img.get_component('c')
        img_range = max(img.data_range) - min(img.data_range)
        rec_Y = rec_Y / img_range
        decisions_Y = self.get_decision_y(kwargs.get('decisions', dict()))
        scale_hat = decisions_Y.get('scale_log')[:, reference_channel_idx:reference_channel_idx + 1, :, :].clone()
        scale_hat = scale_hat.to(dtype=torch.float32)
        (_, _, h, w) = rec_Y.shape
        s_upsampled = torch.nn.functional.interpolate(scale_hat, size=(h, w), mode='nearest')
        mag_list = [mag1, mag2, mag3]
        thr_list = [thr1, thr2, thr3]
        rec_Y = adptive_sharpness(rec_Y, s_upsampled, mag_list, thr_list)
        rec_Y = rec_Y * img_range

        ans = Image.create_from_tensors(rec_Y,
                                        rec_U,
                                        rec_V,
                                        img.data_range,
                                        bit_depth=img.bit_depth,
                                        format=img.format,
                                        color_space='yuv')

        return [ans] + imgs[1:]


def _to_bchw(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a PyTorch tensor image to BCHW format.

    Args:
        tensor (torch.Tensor): image of the form :math:`(*, H, W)`.

    Returns:
        input tensor of the form :math:`(B, C, H, W)`.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(tensor)}")

    if len(tensor.shape) < 2:
        raise ValueError(f"Input size must be a two, three or four dimensional tensor. Got {tensor.shape}")

    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0)

    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)

    if len(tensor.shape) > 4:
        tensor = tensor.view(-1, tensor.shape[-3], tensor.shape[-2], tensor.shape[-1])

    return tensor


def perform_keep_shape_image(f: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """A decorator that enable `f` to be applied to an image of arbitrary leading dimensions `(*, C, H, W)`.

    It works by first viewing the image as `(B, C, H, W)`, applying the function and re-viewing the image as original
    shape.
    """

    @wraps(f)
    def _wrapper(input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if not isinstance(input, torch.Tensor):
            raise TypeError(f"Input input type is not a Tensor. Got {type(input)}")

        if input.numel() == 0:
            raise ValueError("Invalid input tensor, it is empty.")

        input_shape = input.shape
        input = _to_bchw(input)  # view input as (B, C, H, W)
        output = f(input, *args, **kwargs)
        if len(input_shape) == 3:
            output = output[0]

        if len(input_shape) == 2:
            output = output[0, 0]

        if len(input_shape) > 4:
            output = output.view(*(input_shape[:-3] + output.shape[-3:]))

        return output

    return _wrapper


def _blend_one(input1: torch.Tensor, input2: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    r"""Blend two images into one.

    Args:
        input1: image tensor with shapes like :math:`(H, W)` or :math:`(D, H, W)`.
        input2: image tensor with shapes like :math:`(H, W)` or :math:`(D, H, W)`.
        factor: factor 0-dim tensor.

    Returns:
        : image tensor with the batch in the zero position.
    """
    if not isinstance(input1, torch.Tensor):
        raise AssertionError(f"`input1` must be a tensor. Got {input1}.")
    if not isinstance(input2, torch.Tensor):
        raise AssertionError(f"`input1` must be a tensor. Got {input2}.")

    if isinstance(factor, torch.Tensor) and len(factor.size()) != 0:
        raise AssertionError(f"Factor shall be a float or single element tensor. Got {factor}.")
    if factor == 0.0:
        return input1
    if factor == 1.0:
        return input2
    diff = (input2 - input1) * factor
    res = input1 + diff
    if factor > 0.0 and factor < 1.0:
        return res
    return torch.clamp(res, 0, 1)


@perform_keep_shape_image
def adptive_sharpness(input: torch.Tensor, region: torch.tensor, mag_list: List, thr_list: List) -> torch.Tensor:
    if not isinstance(mag_list[0], torch.Tensor):
        mag_list[0] = torch.as_tensor(mag_list[0], device=input.device, dtype=input.dtype)
        mag_list[1] = torch.as_tensor(mag_list[1], device=input.device, dtype=input.dtype)
        mag_list[2] = torch.as_tensor(mag_list[2], device=input.device, dtype=input.dtype)

    kernel = (
            torch.as_tensor([[5, 5, 5], [5, 24, 5], [5, 5, 5]], dtype=input.dtype, device=input.device)
            .view(1, 1, 3, 3)
            .repeat(input.size(1), 1, 1, 1)
            / 64
    )

    degenerate = torch.nn.functional.conv2d(input, kernel, bias=None, stride=1, groups=input.size(1))
    degenerate = torch.clamp(degenerate, 0.0, 1.0)

    # For the borders of the resulting image, fill in the values of the original image.
    mask = torch.ones_like(degenerate)
    padded_mask = torch.nn.functional.pad(mask, [1, 1, 1, 1])
    padded_degenerate = torch.nn.functional.pad(degenerate, [1, 1, 1, 1])
    result = torch.where(padded_mask == 1, padded_degenerate, input)
    factor = torch.ones_like(result)
    factor = torch.where((region >= thr_list[0]) & (region < thr_list[1]), mag_list[0] * factor, factor)
    factor = torch.where((region >= thr_list[1]) & (region < thr_list[2]), mag_list[1] * factor, factor)
    factor = torch.where(region >= thr_list[2], mag_list[2] * factor, factor)
    diff = (input - result) * factor
    res = result + diff
    res = torch.clamp(res, 0, 1)
    return res

