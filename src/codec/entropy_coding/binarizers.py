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

class Binarizers:
    
    @staticmethod
    def encode_unsigned_unary(bypass_coder, data: int) -> None:
        while data > 0:
            bypass_coder.encode(0, max_symbol_value=1)
            data -= 1
        bypass_coder.encode(1, max_symbol_value=1)
    
    @staticmethod
    def decode_unsigned_unary(bypass_coder) -> int:
        ans = 0
        while bypass_coder.decode(1, max_symbol_value=1) == 0:
            ans += 1
        return ans
    
    @staticmethod
    def encode_unsigned_expgolomb_k0(bypass_coder, data: int) -> None:
        data_size = int(np.floor(np.log2(data+1)))
        Binarizers.encode_unsigned_unary(bypass_coder, data_size)
        value = (data + 1) ^ (1 << data_size)
        max_v = 2**data_size-1
        bypass_coder.encode(value, max_symbol_value=max_v)
        
    @staticmethod
    def decode_unsigned_expgolomb_k0(bypass_coder) -> int:
        data_size = Binarizers.decode_unsigned_unary(bypass_coder)
        max_v = 2**data_size-1
        value = int(bypass_coder.decode(1, max_symbol_value=max_v))
        ans = (1 << data_size) + (value - 1)
        return ans
        
    @staticmethod
    def encode_signed_expgolomb_k0(bypass_coder, data: int) -> None:
        abs_data = abs(data)
        sign = 0 if data <= 0 else 1
        value = 2 * abs_data - sign
        Binarizers.encode_unsigned_expgolomb_k0(bypass_coder, value)
        
        
    @staticmethod
    def decode_signed_expgolomb_k0(bypass_coder) -> int:
        value = Binarizers.decode_unsigned_expgolomb_k0(bypass_coder)
        sign = value & 0x1
        ans_data = (value+1) >> 1
        return -ans_data if sign == 0 else ans_data
        