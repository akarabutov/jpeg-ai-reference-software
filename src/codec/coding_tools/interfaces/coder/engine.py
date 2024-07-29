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

from src.codec.entropy_coding import ECSettingContext, HeaderCoder

from ..base import BaseEngine
from .interface import CoderInterface
from .params import CoderParams
from src.codec.common.utils import pop_param_from_dict

class CoderEngine(CoderInterface, BaseEngine):
    def __init__(self, enable_flag_name:str=None, *args, **kwargs):
        self.enable_flag_name = enable_flag_name
        self.use_coding_headers = kwargs.get('use_coding_headers', True)
        self.has_enabled_flag = kwargs.get('has_enabled_flag', False)
        self.signal_enabled_flag = self.has_enabled_flag and self.use_coding_headers and kwargs.get('signal_enabled_flag', True)
        super(CoderEngine, self).__init__(*args, **kwargs)
        self._params_coder_engine = CoderParams(self.has_enabled_flag, enabled=kwargs.get('enabled', 0))
        self._stream_base_comp = kwargs.get('stream_base_comp', None)
        self._stream_header_part = kwargs.get('stream_header_part', 'tool_header')
        
    @property
    def stream_base_comp(self) -> int:
        if self.has_owner():
            return self.owner.get_owner_param('stream_base_comp', 0) if self._stream_base_comp is None else self._stream_base_comp
        else:
            return 0

    def auto_enableflag_detected_value(self) -> bool:
        """ Check wether the enabled flag could be detected automatically without signalling.
        Return None if not, otherwise return the value of the flag
        """
        return None

    # Recursive functions
    def encode_header_recursively(self, ec: HeaderCoder) -> None:
        with self.set_ec_context(ec, self._stream_header_part):
            if self.signal_enabled_flag:
                detected_enable_value = self.auto_enableflag_detected_value()
                if detected_enable_value is None:
                    ec.encode(self.enabled, max_symbol_value=1, name=f'{self.get_tool_url()} enabled' if self.enable_flag_name is None else self.enable_flag_name)
                    self.logger.debug(f'flag: enabled = {self.enabled} in a substream "{self._stream_header_part}" of a component "{self.stream_base_comp}"')
                else:
                    assert detected_enable_value == self.enabled

            if self.enabled:
                if self.use_coding_headers:
                    self.logger.debug(f'encode the tool header to a substream "{self._stream_header_part}" of a component "{self.stream_base_comp}"')
                    self.encode_header(ec)
                self.for_top_level_children(lambda n, x: x.encode_header_recursively(ec), CoderEngine)

    def decode_header_recursively(self, ec: HeaderCoder) -> None:
        with self.set_ec_context(ec, self._stream_header_part):        
            if self.signal_enabled_flag:
                detected_enable_value = self.auto_enableflag_detected_value()
                if detected_enable_value is None:
                    self.enabled = bool(ec.decode([1], max_symbol_value=1, name=f'{self.get_tool_url()} enabled' if self.enable_flag_name is None else self.enable_flag_name) == 1)
                    self.logger.debug(f'flag: enabled = {self.enabled} in a substream "{self._stream_header_part}" of a component "{self.stream_base_comp}"')
                else:
                    self.enabled = detected_enable_value

            if self.enabled:
                if self.use_coding_headers:
                    self.logger.debug(f'decode the tool header from a substream "{self._stream_header_part}" of a component "{self.stream_base_comp}"')
                    self.decode_header(ec)
                    self.init_decoder()
                self.for_top_level_children(lambda n, x: x.decode_header_recursively(ec), CoderEngine)
            else:
                self.init_decoder()

       
    def encode_header(self, ec: HeaderCoder):
        hf = self.header_base_functions()
        if hf is not None:
            hf.enc_f_header(self, ec)
    
    def decode_header(self, ec: HeaderCoder):
        hf = self.header_base_functions()
        if hf is not None:
            hf.dec_f_header(self, ec)

    def set_ec_context(self, ec: "ECModule", stream_part: str, region_idx: int=0):
        return ECSettingContext(ec, stream_part, self.stream_base_comp, region_idx)