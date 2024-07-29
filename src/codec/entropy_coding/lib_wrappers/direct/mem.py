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
from prettytable import PrettyTable 
from .ec_direct import EcLibDirect
from src.codec.common import ExceptionReadHeader

class LHEngine:
    def __init__(self):
        self.bit_count = 0
        
    def write_bits(self, data: np.ndarray, bits_count: int):
        self.bit_count += len(data) * bits_count
        
    def read_bits(self, data_count: int, bits_count: int):
        raise NotImplementedError
    
    def get_pointer_pos(self) -> int:
        return self.bit_count
        

class Mem:
    def __init__(self, mem: np.ndarray, substream_name: str, verbose: bool = False, *args, **kwargs):
        self.engine = LHEngine() if mem is None else EcLibDirect(mem)
        self.table = PrettyTable([substream_name + "()", "Description", "Value"]) if verbose else None
        
        
    def write_bits(self, data: np.ndarray, bits_count: int, name: str = "") -> None:
        if self.table is not None:
            if len(data) > 1:
                self.table.add_row([f"for (int i = 0; i < {len(data)}; ++i)", "", ""])
                self.table.add_row([f"\t{name}", f"u({bits_count})", ""])
            else:
                self.table.add_row([name, f"u({bits_count})", data[0]])
        for d in data:
            self.engine.write_bits(d, bits_count)
                    
                    
                    
    def read_bits(self, data_count: int, bits_count: int, name: str = "") -> np.ndarray:
        ans = np.zeros([data_count], dtype=np.int64)
        for i in range(data_count):
            output_len, output_data = self.engine.read_bits(bits_count)
            if output_len != bits_count:
                raise ExceptionReadHeader()
            ans[i] = output_data
            
        if self.table is not None:
            if data_count > 1:
                self.table.add_row([f"for (int i = 0; i < {data_count}; ++i)", "", ""])
                self.table.add_row([f"\t{name}", f"u({bits_count})", ""])
            else:
                self.table.add_row([name, f"u({bits_count})", ans[0]])            
        return ans
    
    def terminate(self):
        if self.table is not None:
            print(self.table)
                    
    def get_total_bits(self):
        return self.engine.get_pointer_pos()
                