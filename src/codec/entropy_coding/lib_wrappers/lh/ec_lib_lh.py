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
import math
##
import numpy as np

from ..ec_lib_base import ECLibBase
from .custom_prob_wrapper import CustomProbWrapper
from .sgt_prob_wrapper import SgtProbWrapper
from .bypass_prob_wrapper import BypassProbWrapper


class ECLibLH(ECLibBase):
    def __init__(self, collect_cpu_bits=True, *args, **kwargs):
        super(ECLibLH, self).__init__('ECLibLH', *args, **kwargs)
        self.collect_cpu_bits = collect_cpu_bits
        self.excl_ctx_types = []
        self.total_bits = self.total_bits.to(torch.float)

        # prob_wrappers
        self._init_backend(self)
        self._init_prob_wrappers()

    # ###################################################################################################################
    #  __init__ methods
    # ###################################################################################################################
    def _init_backend(self, *args):
        self.backend = self

    def _init_prob_wrappers(self, *args):
        # print('_init_prob_wrappers of ECLibLH')
        self.prob_wrappers = {
            'Custom': CustomProbWrapper(self.backend),
            'Bypass': BypassProbWrapper(self.backend),
            'Sgt': SgtProbWrapper(self.backend)
        }

    # ###################################################################################################################
    # encode/decode methods
    # ###################################################################################################################
    def encode_init(self, output_mem: np.ndarray):
        super().encode_init(output_mem)
        self.output_mem = output_mem
        self.reset_total_bits()

    def encode_term(self) -> int:
        total_bytes = math.ceil(self.total_bits / 8.0)
        if self.output_mem is not None:
            self.output_mem.fill(0)
        return total_bytes

    def decode_init(self, bit_stream=None):
        pass

    def decode_term(self):
        pass

    # ##################################################################################################################
    # service methods
    # ##################################################################################################################
    def exclude_ctx(self, ctx):
        self.excl_ctx_types.append(ctx)

    def update_label_attrs(self, label, bits, freq):
        if self.__check_tensor(bits):
            if (self.total_bits == 0) and (self.total_bits.device != bits.device):
                self.total_bits = self.total_bits.to(bits.device)
            self.total_bits += bits.to(self.total_bits.device)
        if label is not None:
            self.update_label_metrics(label, bits, freq)

    def __check_tensor(self, x):
        flag = True
        if (not self.backend.collect_cpu_bits) and (x.device.type == 'cpu'):
            flag = False
        return flag
