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

from src.codec.common.utils import pop_param_from_dict

from src.codec.coding_tools.interfaces import BaseEngine
from ..lib_wrappers.ec_lib_base import ECLibBase
from .composite_params import ParamsEC


class ECComposite(BaseEngine):
    num_threads_q: int = 1
    num_threads_z: int = 1
    num_threads_r_0: int = 1
    num_threads_r_1: int = 1
    
    def __init__(self, *args, **kwargs):
        super(ECComposite, self).__init__(*args, **kwargs)
        kwargs, factory = pop_param_from_dict(kwargs, 'factory', None)
        self.factory = factory
        self._params_ec = ParamsEC(types=list(factory.keys()))
        
        self.__module = None

    def __repr__(self):
        return ', '.join(
            [f'{x}: {getattr(self, x)}' for x in self._params_ec.get_params_name_list()])

    def __call__(self, *args, **kwargs) -> ECLibBase:
        params_list = self._params_ec.get_params_name_list()
        for k in params_list:
            if k not in kwargs:
                kwargs[k] = getattr(self, k, None)
        return self.factory.create_instance(name=self.type,
                                            *args,
                                            **kwargs)
