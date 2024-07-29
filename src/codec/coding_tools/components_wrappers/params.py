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


class BasicParams(ParamsBase):
    def __init__(self, *args, **kwargs):
        super(BasicParams, self).__init__(*args, **kwargs)

        add_arg = self.add_single_param

        add_arg('chs_hidden',
                default=kwargs.get('chs_hidden', None),
                type=int,
                nargs="+",
                help='Number of channels of hidden layers')
       
        add_arg('skip_first_ds',
                type=int,
                default=1,
                choices=[0, 1],
                help='Skip first downsampling stage')        

        add_arg('skip_depth_step',
                type=int,
                default=1,
                choices=[0, 1],
                help='Skip depth step (in Chroma component)')         
       
        add_arg('ds_atten_module',
                type=int,
                default=1,
                choices=[0, 1],
                help='Use attention module')        

        add_arg('attn',
                type=int,
                default=1,
                choices=[0, 1],
                help='Use attention mechanism')
        
        add_arg('shuffle_in_ls',
                type=int,
                default=1,
                choices=[0, 1],
                help='Use shuffle in the latent space')
        
        add_arg('skip_last_stage',
                type=int,
                default=1,
                choices=[0, 1],
                help='Skip last stage')        


