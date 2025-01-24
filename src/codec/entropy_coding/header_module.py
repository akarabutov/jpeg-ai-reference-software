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
from typing import Union, Tuple
from src.codec.common import Logger
class HeaderCoder:
    def __init__(self, ec: "ECModule"):
        self.ec = ec
        self.logger = None
        self.bits_count = 0
        
    def set_logger(self, logger: Logger):
        self.logger = logger
        
    @property
    def stream_part(self):
        return self.ec.stream_part
    

    @stream_part.setter
    def stream_part(self, value: str):
        self.ec.stream_part = value
        
    @property
    def stream_base_comp(self):
        return self.ec.stream_base_comp


    @stream_base_comp.setter
    def stream_base_comp(self, value: int):     
        self.ec.stream_base_comp = value
        
    @staticmethod
    def convert_values(values: Union[torch.Tensor, int, list]) -> torch.Tensor:
        if isinstance(values, list):
            values = torch.tensor(values)
        elif not isinstance(values, torch.Tensor):
            values = torch.tensor([values])
        return values
        
    @staticmethod
    def check_values(values: torch.Tensor, max_symbol_value: int) -> bool:
        return (torch.logical_and(values >= 0, values <= max_symbol_value)).all()
    
    @staticmethod
    def calc_bits_count(max_value: int) -> int:
        return int(math.ceil(math.log2(max_value+1)))
    
    def get_substream_size(self) -> int:
        return self.ec.get_substream_size()
        
    def encode(self, values: Union[torch.Tensor, int, list], max_symbol_value: int = None, bits_count: int = None, name: str = None) -> None:
        """Bypass encoding of data

        Args:
            value (Union[torch.Tensor, int]): input data
            bits_count (int): count of bits per each symbol
            max_symbol_value (int): maximum possible value of each symbol
            name (str): name of symbol
        """
        values = HeaderCoder.convert_values(values)
        if values.numel() == 0:
            return
        
        assert (max_symbol_value is not None) or (bits_count is not None)
        if bits_count is None:        
            bits_count = HeaderCoder.calc_bits_count(max_symbol_value)
        max_symbol_value_by_bits = (1 << bits_count) -1
        if max_symbol_value is None:
            max_symbol_value = max_symbol_value_by_bits
        if not HeaderCoder.check_values(values, max_symbol_value):
            raise ValueError(f"Some values of the input tensor {values} are out of range [0, {max_symbol_value}]")
        self.bits_count += bits_count
        self.ec.encode_bypass(values, max_symbol_value_by_bits, name)
        if self.logger is not None:
            self.logger.debug(f"Encoded header flag {name} by {bits_count} bits per symbol and with value(s) {values}")
        
    def decode(self, shape: Union[int, Tuple[int]], max_symbol_value: int = None, bits_count: int = None, device: torch.device = "cpu", name: str = None) -> torch.Tensor:
        """Bypass decoding of data

        Args:
            shape (Tuple[int]): shape of the data for decoding
            bits_count (int): count of bits per each symbol
            max_symbol_value (int): maximum possible value of each symbol
            name (str): name of symbol
        """
        assert (max_symbol_value is not None) or (bits_count is not None)
        total_elems = sum(shape) if isinstance(shape, list) else shape
        if total_elems == 0:
            return torch.tensor([])
        
        if bits_count is None:        
            bits_count = HeaderCoder.calc_bits_count(max_symbol_value)        
        max_symbol_value_by_bits = (1 << bits_count) -1
        if max_symbol_value is None:
            max_symbol_value = max_symbol_value_by_bits        
        ans = self.ec.decode_bypass(shape, max_symbol_value_by_bits, device, name)
        if not HeaderCoder.check_values(ans, max_symbol_value):
            raise ValueError(f"Some values of the input tensor {ans} are out of range [0, {max_symbol_value}]")
        if self.logger is not None:
            self.logger.debug(f"Decoded header flag {name} by {bits_count} bits per symbol and with value(s) {ans}")
            
        return ans
        