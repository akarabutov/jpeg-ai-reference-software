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

import torch

from ..base import BaseModule
from ..params import ParamsBase


class ModelEngineParams(ParamsBase):
    def __init__(self, *args, **kwargs):
        super(ModelEngineParams, self).__init__(*args, **kwargs)

        add_arg = self.add_single_param

        add_arg('resume',
                type=str,
                nargs='+',
                default=list(),
                help='List of models to be resumed (after loading checkpoint from ckpt_files')


class ModelMgmtParams(ParamsBase):
    def __init__(self, *args, **kwargs):
        super(ModelMgmtParams, self).__init__(*args, **kwargs)

        add_arg = self.add_single_param

        add_arg('ckpt_model_name', type=str, default=None, help='Name of model with checkpoint')

        add_arg('clipping_mode_opt', type=int, default=0, help='clipping option, 0: normal test, 1: clipping')

        add_arg('ckpt_files',
                type=str,
                nargs='+',
                default=list(),
                help='List of files with checkpoints')

        add_arg('target_device',
                type=str,
                default='inh',
                choices=['gpu', 'cpu', 'inh'],
                help='Target device to be used for this module')        


    def _params_loaded(self, base: BaseModule) -> None:
        super()._params_loaded(base)
        key = getattr(base, 'target_device')
        values = {'gpu': torch.device('cuda'), 'cpu': torch.device('cpu')}
        value = values.get(key)
        setattr(base, 'device', value)

