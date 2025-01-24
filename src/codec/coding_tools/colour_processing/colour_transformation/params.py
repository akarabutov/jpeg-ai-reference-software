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


class ColorTransformParams(ParamsBase):
    def __init__(self, *args, **kwargs):
        super(ColorTransformParams, self).__init__(*args, **kwargs)
        add_arg = self.add_single_param
        
        add_arg('colour_transform_idx', type=int, default=None, help=r'Colour transform index: in a case of 0 the output is RGB, in a case of 1 the output is YUV, in a case of 2 the output has custom colour format. By default, preprocess selects between 0 and 1 automatically based on the input colour format')

        add_arg('colour_transform_offset', type=int, default=[0,0,0], nargs="+", help=r'Offsets for user-defined colour transform', metavar="[0-255]")
        add_arg('colour_transform_matrix', type=int, default=[1,0,0, 0,1,0, 0,0,1], nargs="+", help=r'Matrix for user-defined colour transform', metavar="[0-255]")


class RdoColorTransformParams(ParamsBase):
    def __init__(self, *args, **kwargs):
        super(RdoColorTransformParams, self).__init__(*args, **kwargs)
        add_arg = self.add_single_param

        add_arg('msssim_weight', type=float, default=0.4, help='max numnber of tries to find beta')

        add_arg('use', type=int, default=0, help='To use or not to use bitrate')

        add_arg('size_downscaler',
                type=int,
                default=4,
                help='How much size is decreased to speed up color transform')
        