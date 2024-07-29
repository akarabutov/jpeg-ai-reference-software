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
import torch.nn as nn
import numpy as np
from typing import List


class ECLibBase(nn.Module):
    """
    ECLibBase
    """
    def __init__(self, name, *args, **kwargs):
        super(ECLibBase, self).__init__()
        self.name = name
        self.backend = None
        self.prob_wrappers = dict()
        self.ae_used = False
        self.mem_size = 0

        # attributes
        self.bits_per_label = dict()
        self.likelihoods_per_label = dict()
        self.register_buffer('total_bits', torch.zeros([1], dtype=torch.int32))

    # ##################################################################################################################
    #  __init__ methods
    # ##################################################################################################################
    def _init_backend(self, *args):
        raise NotImplementedError

    def _init_prob_wrappers(self, *args):
        raise NotImplementedError

    # ##################################################################################################################
    #  get/set/reset methods
    # ##################################################################################################################
    def get_name(self):
        return self.name

    def get_wrapper_probs(self):
        return self.prob_wrappers

    def get_bits_per_label(self):
        return self.bits_per_label

    def get_freq_per_label(self):
        return self.likelihoods_per_label

    def get_total_bits(self):
        return self.total_bits

    def reset_label_attrs(self):
        self.bits_per_label.clear()
        self.likelihoods_per_label.clear()

    def reset_total_bits(self):
        self.total_bits.zero_()
        
    def get_mem_size(self) -> int:
        return self.mem_size

    # ##################################################################################################################
    #  encode/decode methods
    # ##################################################################################################################
    def encode_init(self, output_mem: np.ndarray):
        """Initialization for encoding.
        """
        self.mem_size = len(output_mem)
        pass

    def encode_term(self) -> int:
        """Termination for encoding.
        return number of written bits
        """
        raise NotImplementedError

    def decode_init(self, input_mem: np.ndarray=None):
        """Initialization for decoding.
        """
        self.mem_size = len(input_mem)
        pass

    def decode_term(self):
        """Termination for decoding.
        """
        pass
    
    # ##################################################################################################################
    #  service methods
    # ##################################################################################################################
    def update_label_metrics(self, label, bits, freq):
        self._update_label_bits(label, bits)
        self._update_label_freq(label, freq)

    def _update_label_bits(self, label, bits):
        cur_bits = bits
        if label in self.bits_per_label:
            cur_bits += self.bits_per_label[label]

        item = {label: cur_bits}
        self.bits_per_label.update(item)

    def _update_label_freq(self, label, freq):
        cur_freq = freq
        if label in self.likelihoods_per_label:
            cur_freq += self.likelihoods_per_label[label]

        item = {label: cur_freq}
        self.likelihoods_per_label.update(item)
        
    def store_parameters(self, output_dir: str) -> None:
        pass


class ECLibBaseWithThread(ECLibBase):
    def __init__(self, name, *args, **kwargs):
        super(ECLibBaseWithThread, self).__init__(name, *args, **kwargs)
        self.num_threads = kwargs.get('num_threads', 1)
        self.sizes = list()
        self.sizes.append(0)
        
        
    def set_threads_sizes(self, sizes: List[int]) -> None:
        self.num_threads = len(sizes)
        self.sizes = sizes
        
    def get_threads_sizes(self) -> List[int]:
        return self.sizes
