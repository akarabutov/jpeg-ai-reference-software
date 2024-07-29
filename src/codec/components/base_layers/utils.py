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

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.nn.init as init
from typing import Tuple


# #######################################################################################################################
#  Tensor methods about Scale
# #######################################################################################################################
def clip_image(x):
    return torch.clamp(x, -0.996, 0.996)


def clip_image_sgl(x):
    return torch.clamp(x, 0, 255)


def normalize(x):
    return (x - 127.5) / 128


def denormalize(x):
    return x * 128 + 127.5


# ######################################################################################################################
#  Tensor methods about layer
# ######################################################################################################################
def make_layer(block, planes, num_blocks):
    layers = []
    for _ in range(num_blocks):
        layers.append(block(planes))
    return nn.Sequential(*layers)

def get_divider_on_depth(depth: int, skip_depth_step:bool=False) -> int:
    if skip_depth_step and depth >= 4:
        depth = depth-1    
    divider = 2**depth
    return divider

def get_size_on_depth(h: int, w: int, depth: int, skip_depth_step:bool=False) -> Tuple[int, int]:
    divider = get_divider_on_depth(depth, skip_depth_step)
    return int(math.ceil(h / divider)), int(math.ceil(w / divider))

def parse_size_diff(h, w, depth, sd=2, skip_depth_step=False) -> Tuple[int, int]:
    divider = get_divider_on_depth(depth, skip_depth_step)
    h_i, w_i = get_size_on_depth(h, w, depth, skip_depth_step)
    need_size = sd * divider
    h_diff = int(int(math.ceil(h / need_size) * need_size) / divider) - h_i
    w_diff = int(int(math.ceil(w / need_size) * need_size) / divider) - w_i
    return h_diff, w_diff


def cropping_layer(x, h, w, depth, sd=2, skip_depth_step=False):
    h_diff, w_diff = parse_size_diff(h, w, depth, sd, skip_depth_step)
    if h_diff != 0 or w_diff != 0:
        x = x[:, :, :-h_diff, :] if (h_diff != 0) else x
        x = x[:, :, :, :-w_diff] if (w_diff != 0) else x
    return x


def padding_layer(x, h, w, depth, sd=2, skip_depth_step=False):
    h_diff, w_diff = parse_size_diff(h, w, depth, sd, skip_depth_step)
    if h_diff != 0 or w_diff != 0:
        x = functional.pad(x, (0, w_diff, 0, h_diff), mode='replicate')
    return x


def feature_clipping(x, layer, clip_thres, is_clip):
    if is_clip and clip_thres is not None:
        n, _, h, w = x.shape
        x = torch.clamp(x, min=torch.FloatTensor(clip_thres[layer]['MinList']).unsqueeze(1).unsqueeze(2).unsqueeze(0).repeat(n, 1, h, w).to(x.device), 
                         max=torch.FloatTensor(clip_thres[layer]['MaxList']).unsqueeze(1).unsqueeze(2).unsqueeze(0).repeat(n, 1, h, w).to(x.device))
    return x


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
