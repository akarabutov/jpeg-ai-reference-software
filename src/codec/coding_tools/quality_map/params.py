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



class QualityMapParams(ParamsBase):
    """Parameters for coding mode
    """
    def __init__(self, *args, **kwargs):
        super(QualityMapParams, self).__init__(*args, **kwargs)

        add_arg = self.add_single_param
        
        add_arg('block_qp',
                type=int,
                default=8,
                help='4,8,16')
        add_arg('qp_min',
                type=int,
                default=-2,
                help='-4')
        add_arg('qp_max',
                type=int,
                default=2,
                help='2')
        add_arg('adjust_qp',
                type=int,
                default=0,
                choices=[0,1],
                help='Adjust QP')        
        add_arg('ignore_qp_map_bits',
                type=int,
                default=0,
                choices=[0,1],
                help='Use QP map or not')
        add_arg('qp_map_type',
                type=int,
                default=4,
                help='choosing different method to get qp map')
        add_arg('roi_lt_pos_x_list',
                type=str,
                default='100,200',
                help='roi left-top corner x_position')
        add_arg('roi_lt_pos_y_list',
                type=str,
                default='100,200',
                help='roi left-top corner y_position')
        add_arg('roi_wid_list',
                type=str,
                default='64,64',
                help='roi width List')
        add_arg('roi_hei_list',
                type=str,
                default='64,64',
                help='roi height List')
        add_arg('ROI_map_in_file',
                type=str,
                default=None,
                help='ROI map input path')
        add_arg('ROI_map_out_file',
                type=str,
                default=None,
                help='map in latent space output path')
        add_arg('delta_qp',
                type=int,
                default=5,
                help='delta QP max value')
        add_arg('ignor_map_bit',
                type=int,
                default=0,
                help='ignor q map bit')
        add_arg('num_threads', type=int, default=1, help=r"Number of threads in substream")
