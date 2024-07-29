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

import os
import torch
import numpy as np
from src.codec.common import ExceptionReadHeader

from src.codec.coding_tools.interfaces import CoderEngine
from src.codec.entropy_coding.header_module import HeaderCoder
##
from .params import UDIParams


class UDI(CoderEngine):
    def __init__(self, **kwargs):
        super(UDI, self).__init__(has_enabled_flag=False, stream_header_part="udi", **kwargs)
        self.data = None
        self._udi_params = UDIParams()

    def _params_loaded(self) -> None:
        if self.filepath is not None and os.path.exists(self.filepath):
            self.data = np.fromfile(self.filepath, dtype=np.uint8)

    def encode_header(self, ec: HeaderCoder):
        if self.data is not None:
            ec.encode(torch.from_numpy(self.data), max_symbol_value=255, name="UDI_data")
        
    def decode_header(self, ec: HeaderCoder):
        data_size = ec.get_substream_size()
        self.data = ec.decode([data_size], 255, name="UDI_data").numpy().astype(dtype=np.uint8)
        if self.filepath is not None:
            self.data.tofile(self.filepath)
    