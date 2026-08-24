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

__all__ = ['print_sum', 'fake_tensor']


def print_sum(tensor_name, input_tensor):
    """print sum of tensor, only used for debug"""
    if isinstance(input_tensor, torch.Tensor):
        tensor_sum = input_tensor.detach().cpu().numpy().flatten().sum()
        print('\033[1m\033[45;33m {} {} \033[0m\n'.format(tensor_name, tensor_sum))
    else:
        print('\033[1m\033[45;33m {} {} \033[0m\n'.format(tensor_name, 'is not tensor'))


def fake_tensor(N, C, H, W, device):
    np_tensor = np.zeros((N, C, H, W), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            for i in range(H):
                for j in range(W):
                    np_tensor[n, c, i, j] = ((n + 1) * (c + 1) * i * j) % 255
    ret = torch.from_numpy(np_tensor).to(device)
    return ret
