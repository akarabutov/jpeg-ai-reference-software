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
import numpy as np
from .bypass_prob_wrapper import BypassProbWrapper
from ..ec_lib_base import ECLibBase
import numpy as np
from .mem import Mem

class ECLibDirect(ECLibBase):
    def __init__(
            self,
            *args,
            **kwargs):
        """
 
        Args:
            collectors:
            io_stream: bit_stream for input/output
            coder_type:
            lib_name:
            debug:
            debug_start:
        """
        super(ECLibDirect, self).__init__('ECLibDirect', *args, **kwargs)
        self.backend = None
        self.verbose = kwargs.get('verbose', False)
        self.substream_name = kwargs.get('substream_name', None)


 
    # ##################################################################################################################
    #  __init__ methods
    # ##################################################################################################################
    def _init_prob_wrappers(self):
        self.prob_wrappers = {
            'Custom': None,
            'Bypass': BypassProbWrapper(self.backend),
            'Sgt': None,
        }

    # ##################################################################################################################
    #  encode/decode methods    

    def decode_init(self, input_mem: np.ndarray):
        super().decode_init(input_mem)
        self.backend = Mem(input_mem, verbose=self.verbose, substream_name=self.substream_name)
        self._init_prob_wrappers()

    def decode_term(self):
        if self.backend is not None:
            self.backend.terminate()
        self.backend = None

    def encode_init(self, mem: np.ndarray):
        super().encode_init(mem)
        self.backend = Mem(mem, verbose=self.verbose, substream_name=self.substream_name)
        self._init_prob_wrappers()
        
    def encode_term(self) -> int:
        self.total_bits[0] = torch.tensor(self.backend.get_total_bits())
        self.backend.terminate()
        return self.total_bits[0]