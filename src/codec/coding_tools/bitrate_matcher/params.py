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


class BitrateMatcherParams(ParamsBase):
    def __init__(self, *args, **kwargs):
        super(BitrateMatcherParams, self).__init__(*args, **kwargs)
        add_arg = self.add_single_param

        add_arg('target_tool_idx',
                type=int,
                default=-1,
                help='Target tool idx in a case of disabling of the tool')
        add_arg('target_beta_disp_Y',
                type=int,
                default=0,
                help='Target beta displacment Y in a case of disabling of the tool')
        add_arg('target_beta_disp_UV',
                type=int,
                default=None,
                help='Target beta displacment UV in a case of disabling of the tool. It is equal to target_beta_disp_Y by default')

        add_arg('tolerance_max',
                type=float,
                default=0.01,
                help='allowable max(real_bitrate - target_bitrate)/target_bitrate')

        add_arg('tolerance_min',
                type=float,
                default=-0.01,
                help='allowable min(real_bitrate - target_bitrate)/target_bitrate')

        add_arg('beta_min_mult',
                type=float,
                default=1.0/32.0,
                help='minimal multiplicator of beta for the search start')

        add_arg('beta_max_mult', type=float, default=5.0, help='max multiplicator of beta for the search start')

        add_arg('max_iterations_stage1',
                type=int,
                default=20,
                help='max number of tries to find beta')

        add_arg('max_iterations_stage2',
                type=int,
                default=9,
                help='max number of tries to find beta')

        add_arg('max_iterations', type=int, default=20, help='max number of tries to find beta')

        add_arg('bitrate_config_path',
                type=str,
                default='./cfg',
                help='path to a directory with a file with the list of betas')

        add_arg('bitrate_config_name',
                type=str,
                default='betas_4.txt',
                help='name of a file with the list of betas')

        add_arg('use_default',
                type=int,
                default=1,
                help='To use or not to use bitrate matcher if there is not beta list')

        add_arg(
            'regen_beta_list',
            type=int,
            default=0,
            help=
            'Regenerate beta list in a case of bitrate matching. Works only if use_default is 0')

        add_arg(
            'rewrite_beta_list_file',
            type=int,
            default=0,
            help=
            'Overwrite existing beta list file in a case of bitrate matching. Should be used by eval script only.'
        )

        add_arg('independent_beta_UV', type=int, default=0, help='To use or not to use bitrate')

        add_arg('default_models',
                type=int,
                nargs='+',
                default=[0, 1, 2, 3, 4],
                help='default models')
        add_arg('default_target_rates',
                type=int,
                nargs='+',
                default=[12, 25, 50, 75, 100],
                help='default target rates for models')        
        add_arg('default_beta_disp_log',
                type=int,
                nargs='+',
                default=[16, 32, 64, 64, -16],
                help='default target rates for models')               
