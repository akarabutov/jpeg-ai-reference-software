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

class StreamBitWriter():
    def __init__(self, fid, byteorder='big'):
        self.fid = fid
        self.cur_symbol = 0
        self.current_bit = 0
        self.symbols_list = list()
        self.byteorder = byteorder

    @staticmethod
    def calc_precision(max_symbol_value:int) -> int:
        return int(np.ceil(np.log2(max_symbol_value + 1)))    

    def encode(self, value: int, max_symbol_value:int=1) -> None:
        bytes_count = self.calc_precision(max_symbol_value)
        for cn in reversed(range(bytes_count)):
            cur_val = (value >> cn) & 1
            self.current_bit += 1
            self.cur_symbol  <<= 1
            self.cur_symbol = self.cur_symbol | cur_val

            if self.current_bit == 8:
                self.symbols_list.append(self.cur_symbol)
                self.cur_symbol = 0
                self.current_bit = 0

    def get_total_bits(self) -> int:
        return self.current_bit

    def get_total_bytes(self) -> int:
        return (self.get_total_bits() + 7) // 8
    
    def flush(self):
        for s in self.symbols_list:
            self.fid.write(s.to_bytes(1, byteorder=self.byteorder))
        if self.current_bit != 0:
            val = self.cur_symbol << (8-self.current_bit)
            self.fid.write(val.to_bytes(1, self.byteorder))
        self.current_bit = 0
        self.cur_symbol = 0
        self.symbols_list = list()

class StreamBitReader():
    """
    8-bit stream reader
    """
    def __init__(self, fid, byteorder='big'):
        self.fid = fid
        self.cur_symbol = 0
        self.current_bit = 0
        self.byteorder = byteorder
    
    @staticmethod
    def calc_precision(max_symbol_value:int) -> int:
        return int(np.ceil(np.log2(max_symbol_value + 1)))    

    def decode(self, count:int, max_symbol_value:int=1) -> np.array:
        symbol_len = self.calc_precision(max_symbol_value)
        ans = np.zeros([count])
        
        for sn in range(count):
            cur_bits_count = symbol_len
            cur_val = 0
            while cur_bits_count > 0:
                if self.current_bit == 0:
                    self.cur_symbol = int.from_bytes(self.fid.read(1), self.byteorder)
                    self.current_bit = 8
                
                cur_bit = (self.cur_symbol >> (self.current_bit-1)) & 1
                cur_val <<= 1
                cur_val = cur_val | cur_bit
                
                self.current_bit -= 1
                cur_bits_count -= 1
            ans[sn] = cur_val
            
        return ans

    def get_total_bits(self) -> int:
        return self.current_bit

    def get_total_bytes(self) -> int:
        return (self.get_total_bits() + 7) // 8
