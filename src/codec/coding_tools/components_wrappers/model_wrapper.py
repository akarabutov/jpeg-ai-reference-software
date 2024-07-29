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

from src.codec.coding_tools.interfaces import ModelEngine, ParamsBase


class ModelWrapper(ModelEngine):
    def __init__(self, component_type, params: ParamsBase, *args, **kwargs):
        super(ModelWrapper, self).__init__(*args, **kwargs)
        self._params_inst = params
        self._inst_type = component_type
        self.kwargs_init = kwargs
        self._register_load_state_dict_pre_hook(self._load_from_state_dict_hook)
    
    def build_model(self):
        params_dict = self.kwargs_init
        if self._params_inst is not None:
            for k in self._params_inst.get_params_name_list():
                if hasattr(self, k) and k not in params_dict:
                    params_dict[k] = getattr(self, k)
        
        self.inst = self._inst_type(**params_dict)
        
    def forward(self, *args, **kwargs):
        return self.inst(*args, **kwargs)
        
    def _load_from_state_dict_hook(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key_list = list()
        exclude_keys_list = ['epoch', 'best_loss', 'optimizer']
        for k in exclude_keys_list:
            if k in state_dict:
                state_dict.pop(k)
                
        for k in state_dict.keys():
            if k.startswith(prefix):
                key_list.append(k)
            
        for k in key_list:
            new_k = f"{prefix}inst.{k[len(prefix):]}"
            state_dict[new_k] = state_dict.pop(k)
            
def warp_component(component, params: ParamsBase, **kwargs_common):
    def init_model(*args, **kwargs):
        return ModelWrapper(component, params, *args, **kwargs, **kwargs_common)
    return init_model