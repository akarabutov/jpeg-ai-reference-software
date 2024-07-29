import math
import torch
import torch.nn as nn

from src.codec.common import determinism_on_eval
from src.codec.components.base_layers import (conv3x3i, conv1x1i, cropping_layer)
from src.codec.common.utils import is_int_dtype

from ..base import AEBase

class HyperScaleDecoder(AEBase):
    def __init__(self, chs=192, skip_depth_step: bool = False, *args, **kwargs):
        super(HyperScaleDecoder, self).__init__()
        self.skip_depth_step = skip_depth_step

        self.conv1 = conv1x1i(chs, chs, stride=1)

        self.depthwise = conv3x3i(chs, chs, stride=1, groups=1)

        self.pointwise = conv1x1i(chs, chs * 16, stride=1)

        self.relu = nn.ReLU(inplace = True)

        self.ps = torch.nn.PixelShuffle(4)


    @determinism_on_eval
    def forward(self, x, h=0, w=0):
        is_integer_pipeline  = is_int_dtype(x)
        emulate_quantization = is_integer_pipeline and (not self.pointwise.is_quantized)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.depthwise(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.ps(x)
        x = cropping_layer(x, h, w, 4, sd=4, skip_depth_step=self.skip_depth_step)
        if x.dtype == torch.float64:
            x = x.to(dtype=torch.float32)

        if emulate_quantization:
            x = ( x * pow(2, self.sigma_out_precision) ).round().to(torch.int32) #support of non-quantized models
        x = x.abs()

        x = x.clamp(0, self.sigma_idx_max_value)

        return x
