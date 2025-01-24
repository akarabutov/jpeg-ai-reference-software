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

import os
import sys
import numpy
import torch
file_abs_path = os.path.abspath(__file__)
prog_dir = os.path.dirname(os.path.dirname(os.path.dirname(file_abs_path)))
sys.path.append(os.path.join(prog_dir))

sys.path.append(file_abs_path)
from channel_list import sorted_channel_ids_dict

def reorder_channel_wise_entropy(ckpt_name, out_ckpt_name):
    state_dict = torch.load(ckpt_name)
    channel_wise_entropy = state_dict['channel_wise_entropy'].detach().cpu().numpy()
    sorted_ids = (numpy.argsort(channel_wise_entropy)[::-1]).tolist()
    state_dict['channel_wise_entropy'] = state_dict['channel_wise_entropy'][sorted_ids]

    torch.save(state_dict, out_ckpt_name)

ckpt_names = ["Y_0.002.pth", "Y_0.012.pth", "Y_0.075.pth", "Y_0.5.pth"]
model_keys = ["Y_0.002", "Y_0.012", "Y_0.075", "Y_0.5"]
out_ckpt_names = ["Y_0.002.pth", "Y_0.012.pth", "Y_0.075.pth", "Y_0.5.pth"]
for idx in range(len(ckpt_names)):
    ckpt_name = ckpt_names[idx]
    out_ckpt_name = out_ckpt_names[idx]
    reorder_channel_wise_entropy(ckpt_name, out_ckpt_name)

ckpt_names = ["UV_0.002.pth", "UV_0.012.pth", "UV_0.075.pth", "UV_0.5.pth"]
model_keys = ["UV_0.002", "UV_0.012", "UV_0.075", "UV_0.5"]
out_ckpt_names = ["UV_0.002.pth", "UV_0.012.pth", "UV_0.075.pth", "UV_0.5.pth"]
for idx in range(len(ckpt_names)):
    ckpt_name = ckpt_names[idx]
    out_ckpt_name = out_ckpt_names[idx]
    reorder_channel_wise_entropy(ckpt_name, out_ckpt_name)
