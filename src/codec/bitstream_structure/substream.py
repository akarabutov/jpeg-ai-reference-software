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
from io import FileIO, BytesIO
from src.codec.entropy_coding import StreamBitWriter, StreamBitReader, Binarizers
from .layouts_def import SubstreamLayouts
        

    
class Substream:
    """Class for storing data of each substream
    """
    def __init__(self, byte_order:str="big", marker_id: int = None, mem_stream: BytesIO = None):
        self.marker_id = marker_id
        self.mem_stream = mem_stream
        self.BYTE_ORDER = byte_order
        
        
    @staticmethod
    def encode_marker_id(fid: FileIO, marker_id: int, byte_order:str="big") -> None:
        fid.write(marker_id.to_bytes(2, byte_order))
        
        
    def __encode_substream_header(self, fid: FileIO, marker_id: int, substream_size: int) -> int:
        """Encode header of substream

        Args:
            fid (FileIO): file/memory descriptior for storing data
            marker_id (int): marker ID of the substream
            substream_size (int): size of the substream

        Returns:
            int: total size of the header
        """
        ans = 0
        self.encode_marker_id(fid, marker_id, self.BYTE_ORDER)
        bypass_codec = StreamBitWriter(fid, byteorder=self.BYTE_ORDER)
        Binarizers.encode_unsigned_expgolomb_k0(bypass_codec, substream_size)
        ans = 2 + bypass_codec.get_total_bytes()
        bypass_codec.flush()
        return ans
        
        
    def __decode_substream_size(self, fid: FileIO) -> int:
        """Decode size of the substream

        Args:
            fid (FileIO): file/memory descriptior for reading of data

        Returns:
            int: size of the substream
        """
        bypass_codec = StreamBitReader(fid, self.BYTE_ORDER)
        substream_size = int(Binarizers.decode_unsigned_expgolomb_k0(bypass_codec))
        return substream_size        
        
        
    def get_substream_size(self) -> int:
        """Get substream size (except region idx if it has)

        Returns:
            int: _description_
        """
        return len(self.mem_stream.getvalue())
        
        
    def read(self, fid: FileIO) -> bool:
        """Read substream from bitstream

        Args:
            fid (FileIO): file descriptior
            may_have_regID (bool, optional): flag that the substream may have region ID if the substream supports regions. Defaults to False.

        Returns:
            bool: substream was read succesfully
        """
        self.marker_id = int.from_bytes(fid.read(2), self.BYTE_ORDER)
        found = False
        for s in SubstreamLayouts.get_substream_type_gen():
            if s.marker_id == self.marker_id:
                found = True
                self.substream_size = self.__decode_substream_size(fid)
                self.mem_stream = BytesIO()
                self.mem_stream.write(fid.read(self.substream_size))
                break
        return found
        
        
    def write(self, fid: FileIO) -> None:
        """Store substream to a file

        Args:
            fid (FileIO): _description_
        """
        self.mem_stream.seek(0)
        self.__encode_substream_header(fid, self.marker_id, len(self.mem_stream.getvalue()))
        fid.write(self.mem_stream.getvalue())
        
        