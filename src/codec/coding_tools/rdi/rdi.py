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
from .params import RDIParams

class RDI(CoderEngine):
    def __init__(self, **kwargs):
        super(RDI, self).__init__(has_enabled_flag=False, 
                                  stream_header_part="rdi", **kwargs)
        self._rdi_params = RDIParams()

    def _params_loaded(self) -> None:
        self.cicp_info_present_flag = int(self.cicp_info_present_flag)
        self.mdcv_info_present_flag = int(self.mdcv_info_present_flag)
        self.clli_info_present_flag = int(self.clli_info_present_flag)
        self.dm_present_flag        = int(self.dm_present_flag)
        self.dm_size = (self.dm_data_byte.bit_length()+7)//8

    def encode_header(self, ec: HeaderCoder):
        # encode present flags
        ec.encode(self.cicp_info_present_flag, max_symbol_value=1, 
                  name="cicp_info_present_flag")
        ec.encode(self.mdcv_info_present_flag, max_symbol_value=1, 
                  name="mdcv_info_present_flag")
        ec.encode(self.clli_info_present_flag, max_symbol_value=1, 
                  name="clli_info_present_flag")
        ec.encode(self.dm_present_flag,        max_symbol_value=1, 
                  name="dm_present_flag")
        ec.encode(0,        max_symbol_value=15, 
                  name="ri_reserved_zero_4bits")

        if self.cicp_info_present_flag:
            ec.encode(self.colour_primaries,
                      max_symbol_value=2 ** 8 - 1,
                      name='colour_primaries')
            ec.encode(self.transfer_characteristics,
                      max_symbol_value=2 ** 8 - 1,
                      name='transfer_characteristics')
            ec.encode(self.matrix_coefficients,
                      max_symbol_value=2 ** 8 - 1,
                      name='matrix_coefficients')
            ec.encode(self.image_full_range_flag,
                      max_symbol_value=1,
                      name='image_full_range_flag')
            ec.encode(self.chroma420_sample_loc_type,
                      max_symbol_value=2 ** 3 - 1,
                      name='chroma420_sample_loc_type')
            ec.encode(0,        max_symbol_value=15, 
                      name="ri_reserved_zero_4bits")
        if self.mdcv_info_present_flag:
            ec.encode(self.mastering_display_colour_primaries_x[0],
                      max_symbol_value=2 ** 16 - 1,
                      name='mastering_display_colour_primaries_x_0')
            ec.encode(self.mastering_display_colour_primaries_y[0],
                      max_symbol_value=2 ** 16 - 1,
                      name='mastering_display_colour_primaries_y_0')
            ec.encode(self.mastering_display_colour_primaries_x[1],
                      max_symbol_value=2 ** 16 - 1,
                      name='mastering_display_colour_primaries_x_1')
            ec.encode(self.mastering_display_colour_primaries_y[1],
                      max_symbol_value=2 ** 16 - 1,
                      name='mastering_display_colour_primaries_y_1')
            ec.encode(self.mastering_display_colour_primaries_x[2],
                      max_symbol_value=2 ** 16 - 1,
                      name='mastering_display_colour_primaries_x_2')
            ec.encode(self.mastering_display_colour_primaries_y[2],
                      max_symbol_value=2 ** 16 - 1,
                      name='mastering_display_colour_primaries_y_2')
            ec.encode(self.mastering_display_white_point_chromaticity_x,
                      max_symbol_value=2 ** 16 - 1,
                      name='mastering_display_white_point_chromaticity_x')
            ec.encode(self.mastering_display_white_point_chromaticity_y,
                      max_symbol_value=2 ** 16 - 1,
                      name='mastering_display_white_point_chromaticity_y')
            ec.encode(self.mastering_display_maximum_luminance,
                      max_symbol_value=2 ** 32 - 1,
                      name='mastering_display_maximum_luminance')
            ec.encode(self.mastering_display_minimum_luminance,
                      max_symbol_value=2 ** 32 - 1,
                      name='mastering_display_minimum_luminance')
        if self.clli_info_present_flag:
            ec.encode(self.maximum_content_light_level,
                      max_symbol_value=2 ** 16 - 1,
                      name='maximum_content_light_level')
            ec.encode(self.maximum_frame_average_light_level,
                      max_symbol_value=2 ** 16 - 1,
                      name='maximum_content_light_level')
        if self.dm_present_flag:
            ec.encode(self.dm_type,
                      max_symbol_value=2 ** 8 - 1, name='dm_type')
            ec.encode(self.dm_size,
                      max_symbol_value=2 ** 16 - 1, name='dm_size')
            ec.encode(self.dm_data_byte,
                      max_symbol_value=2**(self.dm_size*8)-1,
                      name='dm_data_byte')

    def decode_header(self, ec: HeaderCoder):
        # decode present flags
        self.cicp_info_present_flag = int(ec.decode([1],
                    max_symbol_value=1, name='cicp_info_present_flag'))
        self.mdcv_info_present_flag = int(ec.decode([1],
                    max_symbol_value=1, name='mdcv_info_present_flag'))
        self.clli_info_present_flag = int(ec.decode([1],
                    max_symbol_value=1, name='clli_info_present_flag'))
        self.dm_present_flag        = int(ec.decode([1],
                    max_symbol_value=1, name='dm_present_flag'))
        tmp                         = int(ec.decode([1], 
                    max_symbol_value=15, name='ri_reserved_zero_4bits'))
        assert(tmp == 0)

        if self.cicp_info_present_flag:
            self.colour_primaries = int(ec.decode([1],
                                    max_symbol_value=2 ** 8 - 1,
                                    name="colour_primaries"))
            self.transfer_characteristics = int(ec.decode([1],
                                    max_symbol_value=2 ** 8 - 1,
                                    name="transfer_characteristics"))
            self.matrix_coefficients = int(ec.decode([1],
                                    max_symbol_value=2 ** 8 - 1,
                                    name="matrix_coefficients"))
            self.image_full_range_flag = int(ec.decode([1],
                                    max_symbol_value=1, 
                                    name='image_full_range_flag'))
            self.chroma420_sample_loc_type = int(ec.decode([1],
                                    max_symbol_value=2 ** 3 - 1,
                                    name="chroma420_sample_loc_type"))
            tmp                         = int(ec.decode([1], 
                                    max_symbol_value=15, name='ri_reserved_zero_4bits'))
            assert(tmp == 0)

        if self.mdcv_info_present_flag:
            for i in range(3):
                self.mastering_display_colour_primaries_x[i]=int(ec.decode([1],
                            max_symbol_value=2 ** 16 - 1,
                            name=f'mastering_display_colour_primaries_x_{i}'))
                self.mastering_display_colour_primaries_y[i]=int(ec.decode([1],
                            max_symbol_value=2 ** 16 - 1,
                            name=f'mastering_display_colour_primaries_y_{i}'))
            self.mastering_display_white_point_chromaticity_x = int(ec.decode([1],
                            max_symbol_value=2 ** 16 - 1,
                            name='mastering_display_white_point_chromaticity_x'))
            self.mastering_display_white_point_chromaticity_y = int(ec.decode([1],
                            max_symbol_value=2 ** 16 - 1,
                            name='mastering_display_white_point_chromaticity_y'))
            self.mastering_display_maximum_luminance = int(ec.decode([1],
                            max_symbol_value=2 ** 32 - 1,
                            name='mastering_display_maximum_luminance'))
            self.mastering_display_minimum_luminance = int(ec.decode([1],
                            max_symbol_value=2 ** 32 - 1,
                            name='mastering_display_minimum_luminance'))
        if self.clli_info_present_flag:
            self.maximum_content_light_level = int(ec.decode([1],
                            max_symbol_value=2 ** 16 - 1,
                            name="maximum_content_light_level"))
            self.maximum_frame_average_light_level = int(ec.decode([1],
                            max_symbol_value=2 ** 16 - 1,
                            name="maximum_frame_average_light_level"))
        if self.dm_present_flag:
            self.dm_type = int(ec.decode([1],
                                max_symbol_value=2 ** 8 - 1,
                                name="dm_type"))
            self.dm_size = int(ec.decode([1],
                                max_symbol_value=2 ** 16 - 1,
                                name="dm_size"))
            self.dm_data_byte = int(ec.decode([1],
                                max_symbol_value=2**(self.dm_size*8)-1,
                                name="dm_data_byte"))
