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
 
import ctypes
import numpy as np
import torch
 
 
class BaseProbWrapper:
    range_checked = True
 
    def __init__(self, backend=None):
        """
 
        Args:
            backend: object newed by 'ctypes.cdll.LoadLibrary'
        """
        super(BaseProbWrapper, self).__init__()
        self.backend = backend
 
    # ##################################################################################################################
    #  encode/decode methods
    # ##################################################################################################################
    def encode(self, *args, **kwargs):
        raise NotImplementedError
 
    def decode(self, *args, **kwargs):
        raise NotImplementedError
 
    # ##################################################################################################################
    #  service methods
    # ##################################################################################################################
    @staticmethod
    def check_range(x, min, max):
        flag = ((x >= min) & (x <= max)).all()
        if not flag:
            raise ValueError('x is not in [{}, {}]'.format(min, max))
 
    def recurant_list_conv(x: list) -> list:
        ans = list()
        for k in x:
            if isinstance(k, list):
                ans += BaseProbWrapper.recurant_list_conv(k)
            else:
                ans += BaseProbWrapper.convert_data(k)
        return ans
 
    @staticmethod
    def convert_data(x, dtype = None):
        
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, list):
            return torch.tensor(x, dtype = dtype)
        elif isinstance(x, bool) or isinstance(x, int) or isinstance(x, float):
            return torch.tensor([x], dtype = dtype)
        else:
            raise NotImplementedError
 
    @staticmethod
    def new_output(shape, ctype=ctypes.c_float):
        size = int(np.prod(shape))
        cptr = (ctype * size)()
        # print('new_data: cptr={}, size={}.'.format(cptr, size))
        return cptr, size
 
    @staticmethod
    def input_to_ctype(x, dtype=np.uint32):
        size = int(np.prod(x.shape))
        ndarr = x.squeeze(0).cpu().numpy().reshape(size).astype(dtype, copy=False)
        cptr = ndarr.ctypes.data_as(ctypes.c_char_p)
        # print('parse_data: cptr={}, size={}.'.format(cptr, size))
        return cptr, size
 
    @staticmethod
    def output_from_ctype(cptr, shape, dtype=np.uint32, device = None):
        x = np.array(cptr[:], dtype)
        
        if dtype == np.uint32:
            x = x.astype(np.int32, copy=False)
        
        x = torch.tensor(x, device = device)
        x = x.view(shape)
        return x
