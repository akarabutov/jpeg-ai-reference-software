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

from ...interfaces import ParamsBase


class IcciParams(ParamsBase):
    def __init__(self, *args, **kwargs):
        super(IcciParams, self).__init__(*args, **kwargs)
        add_arg = self.add_single_param

        add_arg('loss_type', type=str, default='mse', choices=['mse', 'ms-ssim', 'mixed'], help='')
        add_arg('luma_loss_weights', type=float, nargs="+", default=[7., 1.], help='')
        add_arg('chroma_loss_weights', type=float, nargs="+", default=[1., 0.], help='')

        add_arg('in_nc',  default=1, type=int,  help='input number channel')
        add_arg('out_nc', default=1, type=int,  help='output number channel')
        add_arg('nf',     default=48, type=int, help='ICCI number channel')
        add_arg('nbY',    default=2, type=int, help='luma layer number')
        add_arg('nbUV',   default=2, type=int, help='chroma layer number')
        add_arg('process_short_list', type=int,   default=1, help='process only filters from the short list')
        
        default_y_short_list = {
                                "sop":
                                    {0: [5, 6], 1: [2, 6], 2: [2, 7], 3: [3, 8], 4: [3, 4]},
                                "bop":
                                    {0: [5, 6], 1: [3, 6], 2: [2, 7], 3: [3, 8], 4: [4, 9]},
                                "hop":
                                    {0: [5, 6], 1: [6, 8], 2: [2, 7], 3: [3, 9], 4: [4, 9]}   
                                }

        default_uv_short_list = {
                                "sop":
                                    {0: [0, 1], 1: [0, 1], 2: [1, 2], 3: [1, 2], 4: [3, 4]},
                                "bop":
                                    {0: [0, 1], 1: [1, 2], 2: [2, 3], 3: [2, 3], 4: [1, 4]},
                                "hop":
                                    {0: [0, 2], 1: [1, 3], 2: [2, 3], 3: [2, 3], 4: [3, 4]}   
                                }
        add_arg('y_short_list', default=default_y_short_list, help='short list for y channel')
        add_arg('uv_short_list', default=default_uv_short_list, help='short list for uv channel')
