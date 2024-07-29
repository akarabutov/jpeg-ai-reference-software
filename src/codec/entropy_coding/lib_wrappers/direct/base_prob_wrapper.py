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


class BaseProbWrapper:
    def __init__(self, backend):
        super(BaseProbWrapper, self).__init__()
        self.backend = backend

    # ##################################################################################################################
    #  encode/decode methods
    # ##################################################################################################################
    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        # print('>>> Proxy/BaseProbWrapper: decode', flush=True)
        # print('<<< Proxy/BaseProbWrapper: decode', flush=True)
        pass

    # ##################################################################################################################
    #  service methods
    # ##################################################################################################################
    def compute_bits(self, *args, **kwargs):
        raise NotImplementedError

    def compute_freq(self, *args, **kwargs):
        raise NotImplementedError

    # ##################################################################################################################
    #  static methods
    # ##################################################################################################################
    @staticmethod
    def get_size(x: torch.Tensor):
        return x.numel()

    @staticmethod
    def compute_symbol_bits(symbol_num, device, dtype):
        tensor = torch.tensor(symbol_num, device=device, dtype=dtype)
        bits = torch.log2(tensor)
        return bits

    @staticmethod
    def convert_data(x):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, list):
            return torch.Tensor(x)
        elif isinstance(x, int) or isinstance(x, float):
            return torch.Tensor([x])
        else:
            raise NotImplementedError
