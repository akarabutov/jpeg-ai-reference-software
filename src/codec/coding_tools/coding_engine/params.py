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

from ..interfaces import ParamsBase


class AicParams(ParamsBase):
    def __init__(self, *args, **kwargs):
        super(AicParams, self).__init__(*args, **kwargs)
        add_arg = self.add_single_param

        add_arg('target_bpps', type=int, nargs='+', default=[3, 2, 1, 0])
        add_arg('entropy_scale_bound', type=float, default=1E-9)

        add_arg('c_ver_value',
                type=int,
                default=None,
                choices=[1,2],
                help='Predefined value of "c_ver". By default it is None, that means the value will be determined based on s_ver')
        add_arg('c_hor_value',
                type=int,
                default=None,
                choices=[1,2],
                help='Predefined value of "c_hor". By default it is None, that means the value will be determined based on s_hor')

        add_arg('diff_display_img_width', type=int, default=0, help=r"diff_display_img_width is a value ranging from 0 to 63; display_image_width = img_width – diff_display_img_width display_image_width specifies that pixels in the column 0 to column display_image_width - 1 of the decoded image are for display, and the pixels in the rest of the columns are not for display;")
        add_arg('diff_display_img_height', type=int, default=0, help=r"diff_display_img_height is a value ranging from 0 to 63; display_image_height = img_height – diff_display_img_height display_image_height specifies that pixels in the row 0 to row display_image_height - 1 of the decoded image are for display, and the pixels in the rest of the rows are not for display;") 
        
        add_arg('synthesis_transform_id', type=int, nargs="+", default=[0,1,2], help="Supported decoder indexes")
        add_arg('stream_profile_id', type=int, default=0, choices=[0], help='Specifies stream profile, 0 means.')
        add_arg('decoder_profile_id', type=int, default=0, choices=[0,1,2], help='Specifies decoder profile. Only one profile is supported: 0.')
        add_arg('level_idc', type=int, default=52, choices=[10,11,12,20,21,22,30,31,32,40,41,42,50,51,52], help='Indicates the level to which the codestream conforms.')
        add_arg('decoder_id', type=int, default=None, help="Default decoder ID, which will be used to select synthesis network. The value should be in a list of 'synthesis_transform_id'.")