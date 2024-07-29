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

class QuantizerParams(ParamsBase):
    def __init__(self, *args, **kwargs):
        super(QuantizerParams, self).__init__(*args, **kwargs)

        add_arg = self.add_single_param
        add_arg('beta_displacement_precision',
                type=int,
                default=5,
                help=r'Precision of beta_displacement')
        add_arg('beta_displacement_log_bitdepth',
                type=int,
                default=12,
                help=r'Bit-depth of beta_displacement_log')
        add_arg('sigma_precision',
                type=int,
                default=7,
                help=r'Precision of sigma')        
        add_arg('gain_vector_bitdepth',
                type=int,
                default=6,
                help=r'Bit-depth of gain vector')   
        add_arg('gain_vector_precision',
                type=int,
                default=5,
                help=r'Gain vector precision')   
        add_arg('gain_vector_log_bitdepth',
                type=int,
                default=12,
                help=r'Bit-depth of gain vector in log domain')
        add_arg('beta_displacement_log_low_bound',
                type=int,
                default=-1069,
                help=r'Lower boundary of beta_displacement_log clipping')
        add_arg('beta_displacement_log_high_bound',
                type=int,
                default=702,
                help=r'Higher boundary of beta_displacement_log clipping')
