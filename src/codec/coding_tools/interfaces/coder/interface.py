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

from src.codec.common import Decisions
from src.codec.entropy_coding import ECModule, HeaderCoder
from .header_proxy import HeaderBaseFuncs

class CoderInterface:
    def __init__(self, *args, **kwargs):
        super(CoderInterface, self).__init__(*args, **kwargs)
        
    @staticmethod
    def header_base_functions() -> HeaderBaseFuncs:
        """Virtual function for returning of a set of base functions for reading/writing headers
        """
        return None        

    # The second stage on encoder-side: encoding latent information to bitstream
    def encode(self, ec: ECModule, decision: Decisions, *args, **kwargs) -> None:
        """
        Encode decisions to bitstream
        """
        raise NotImplementedError

    # The first stage on decoder-side: decoding latent information from bitstream
    def decode(self, ec: ECModule, *args, **kwargs) -> Decisions:
        """
        Decode decision from bitstream
        """
        raise NotImplementedError

    # Encoding header information if the tool is on
    def encode_header(self, ec: HeaderCoder):
        pass

    # Decoding header information if the tool is on
    def decode_header(self, ec: HeaderCoder):
        pass

    # Initialize module if decode_header wasn't called
    def init_decoder(self):
        pass
