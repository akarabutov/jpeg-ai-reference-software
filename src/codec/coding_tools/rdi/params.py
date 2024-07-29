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

from src.codec.coding_tools.interfaces import ParamsBase

class RDIParams(ParamsBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        add_arg = self.add_single_param

        add_arg('cicp_info_present_flag', type=int, default=0, choices=[0,1],
                help='Flag to signal if CICP rendering info is present')
        add_arg('mdcv_info_present_flag', type=int, default=0, choices=[0,1],
                help='Flag to signal if MDCV rendering info is present')
        add_arg('clli_info_present_flag', type=int, default=0, choices=[0,1],
                help='Flag to signal if CLLI rendering info is present')
        add_arg('dm_present_flag', type=int, default=0, choices=[0,1],
                help='Flag to signal if dynamic metadata is present')
        add_arg('colour_primaries', type=int, default=2,
                help='RDI: colour primaries as in CICP')
        add_arg('transfer_characteristics', type=int, default=2,
                help='RDI: transfer characteristics as in CICP')
        add_arg('matrix_coefficients', type=int, default=2,
                help='RDI: matrix coefficients as in CICP')
        add_arg('image_full_range_flag', type=int, default=1,
                help='RDI: same as full range flag in CICP')
        add_arg('chroma420_sample_loc_type', type=int, default=2,
                help='RDI: Chroma 420 Sample Location Type as in CICP')
        add_arg('mastering_display_colour_primaries_x',
                default = [5,5,5], type=int, nargs=3,
                help='Mastering Display Colour Primaries x')
        add_arg('mastering_display_colour_primaries_y',
                default = [5,5,5], type=int, nargs=3,
                help='Mastering Display Colour Primaries y')
        add_arg('mastering_display_white_point_chromaticity_x', 
                type=int, default=5,
                help='Mastering Display White Point Chromaticity x')
        add_arg('mastering_display_white_point_chromaticity_y', 
                type=int, default=5,
                help='Mastering Display White Point Chromaticity y')
        add_arg('mastering_display_maximum_luminance',
                type=int, default=50000,
                help='Mastering Display Maximum Luminance')
        add_arg('mastering_display_minimum_luminance',
                type=int, default=1,
                help='Mastering Display Minimum Luminance')
        add_arg('maximum_content_light_level',
                type=int, default=1000,
                help='Maximum Content Light Level')
        add_arg('maximum_frame_average_light_level',
                type=int, default=100,
                help='Maximum Frame Average Light Level')
        add_arg('dm_type', type=int, default = 0,
                help='Dynamic metadata type for rendering information')
        add_arg('dm_size', type=int, default=2,
                help='The number of bytes of the dynamic metadata')
        add_arg('dm_data_byte', type=int, default=1234,
                help='Dynamic metadata payload')