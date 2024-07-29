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

##
import math
import numpy as np
from typing import List, Union
from io import BytesIO
from src.codec.entropy_coding.lib_wrappers.ec_lib_base import ECLibBase, ECLibBaseWithThread
from src.codec.entropy_coding import StreamBitWriter, StreamBitReader, Binarizers


class AEMemObject:
    def __init__(self, ae: Union[ECLibBase, ECLibBaseWithThread], mem_size: int, offsets_count: int = None, verbose: bool = False):
        self.ae = ae
        self.verbose = verbose
        self.mem = None if mem_size is None else np.zeros(mem_size, dtype=np.uint8)
        #self.offsets = [] if offsets_count is None else [0] * offsets_count
        self._threads_sizes = [] if offsets_count is None else [0] * (offsets_count-1)
        self.mem_is_ready = False
        
    def get_mem_size(self):
        assert self.mem is not None
        return len(self.mem)
        
    def is_used(self) -> bool:
        """Flag that this object was used

        Returns:
            bool: _description_
        """
        return self.ae.ae_used
        
    def num_threads(self) -> int:
        """Number of threads

        Returns:
            int: number of threads
        """
        return len(self._threads_sizes)+1
    
    def __get_threads_sizes(self) -> List[int]:
        """List of threads sizes

        Returns:
            List[int]: list of threads sizes
        """
        last_thread_size = len(self.mem) - sum(self._threads_sizes )
        positions = self._threads_sizes + [last_thread_size]
        return positions
    
    def _calc_mean_threads_size(self) -> int:
        return int(math.floor(len(self.mem)) / (len(self._threads_sizes) + 1))
    
    
    def __encode_threads_deltas(self, fid: BytesIO, byte_order: str = 'big') -> None:
        """Encode threads deltas

        Args:
            fid (FileIO): file descriptior
            byte_order (str, optional): Byte order in the stream. Defaults to 'big'.
        """
        bypass_codec = StreamBitWriter(fid, byteorder=byte_order)
        mean_thread_size = self._calc_mean_threads_size()
        for ts in self._threads_sizes:
            Binarizers.encode_signed_expgolomb_k0(bypass_codec, mean_thread_size - ts)
        bypass_codec.flush()
        
        
    def __decode_threads_deltas(self, fid: BytesIO, byte_order: str = 'big'):
        """Decode size of the substream's threads

        Args:
            fid (FileIO): file/memory descriptior for reading of data

        Returns:
            int: size of the substream's threads
        """
        bypass_codec = StreamBitReader(fid, byte_order)
        for i in range(len(self._threads_sizes)):
            # TODO: change this part to se(v)
            self._threads_sizes[i] = int(Binarizers.decode_signed_expgolomb_k0(bypass_codec)) 
            
    def __postupdate_threads_deltas(self) -> None:
        mean_thread_size = self._calc_mean_threads_size()
        for i in range(len(self._threads_sizes)):
            # TODO: change this part to se(v)
            self._threads_sizes[i] = mean_thread_size - self._threads_sizes[i]
        
            
    def parse_substream(self, mid: BytesIO, byte_order: str = 'big') -> None:
        """Parse substream 

        Args:
            mid (BytesIO): memory with the substream
            byte_order (str, optional): byte order in the bitstream. Defaults to 'big'.

        Raises:
            NotImplementedError: used unknown type of AE
        """
        
        if isinstance(self.ae, ECLibBaseWithThread):
            self.__decode_threads_deltas(mid, byte_order)
        self.mem = np.frombuffer(mid.read(), dtype=np.uint8)
        self.__postupdate_threads_deltas()
        if isinstance(self.ae, ECLibBaseWithThread):
            self.ae.set_threads_sizes(self.__get_threads_sizes())
        self.ae.decode_init(self.mem)  
        self.ae.ae_used = True
       
        
    def decode_term(self) -> None:
        """Terminate decoder
        """
        self.ae.decode_term()
        self.mem_is_ready = True
        if self.verbose:
            print(f"\tMemory for with len {len(self.mem)} and threads count {len(self._threads_sizes) + 1}")
        
    def encode_init(self) -> None:
        """Initialize encoder
        """
        self.ae.encode_init(self.mem)
    
    def encode_term(self) -> None:
        if not self.mem_is_ready:
            self.ae.encode_term()
            mem_len = (int(self.ae.get_total_bits().item()) + 7) // 8
            self.mem = self.mem[:mem_len]
        self.mem_is_ready = True

    def store_substream(self, mid: BytesIO, byte_order: str = 'big') -> None:
        """Store memory to substream

        Args:
            mid (BytesIO): output memory with substream
            byte_order (str, optional): byte order in the bitstream. Defaults to 'big'.
        """
        self.encode_term()
        if isinstance(self.ae, ECLibBaseWithThread):
            self._threads_sizes = self.ae.get_threads_sizes()[:-1]
            self.__encode_threads_deltas(mid, byte_order)
        else:
            self.offsets =  [0]

        mid.write(self.mem.tobytes())
        if self.verbose:
            print(f"\tMemory with len {len(self.mem)} and threads count {len(self.__get_threads_sizes())}")
        
        