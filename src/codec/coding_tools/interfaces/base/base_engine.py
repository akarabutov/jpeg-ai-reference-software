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

from copy import deepcopy
import os
from typing import Any, Dict

from src.codec.common import ArgParserDecorator

from ..attrs import AttrsProxy, AttrsProxyComposite
from ..params import LoggerParams, ParamsBase, ParamsComposite
##
from .base_module import BaseModule
import math


class BaseEngine(BaseModule):
    def __init__(self, *args, **kwargs):
        super(BaseEngine, self).__init__(*args, **kwargs)
        self._params = ParamsComposite(self)
        self._params_logger = LoggerParams()
        self._attrs_proxies = AttrsProxyComposite(source_obj=self)
        self.enabled = 1

    def is_enabled(self):
        return self.enabled

    def set_enable(self, state):
        self.enabled = state

    def __setattr__(self, name: str, value) -> None:
        if isinstance(value, ParamsBase):
            # Aggregate all parameters in composer
            self.__dict__['_params'].append(value)
        elif isinstance(value, AttrsProxy):
            self.__dict__['_attrs_proxies'].append(value)
        super(BaseEngine, self).__setattr__(name, value)

    @property
    def params(self) -> ParamsComposite:
        return self._params

    def get_params_list_recursively(self, parser: ArgParserDecorator) -> None:
        self.params.def_params_list(parser)
        self.for_top_level_children(
            lambda n, m: m.get_params_list_recursively(parser.add_sub_section_parser(n)))
        
    def set_defval2attrs_recursively(self) -> None:
        self._params.set_defval2attrs()
        self.for_top_level_children(
            lambda n, m: m.set_defval2attrs_recursively())
        

    def load_params2attrs_recursively(self, **params) -> None:
        if len(params) == 0:
            return
        self._params.load_params2attrs(**params)
        self._attrs_proxies.process()
        self.for_top_level_children(
            lambda n, m: m.load_params2attrs_recursively(**params.get(n, {})))

    def store_attrs2dict_recursively(self) -> Dict[str, Any]:
        ans = self._params.store_attrs2dict()
        self.for_top_level_children(lambda n, m: ans.update({n: m.store_attrs2dict_recursively()}))
        return ans
    
    def signal_params_loaded_recursively(self) -> None:
        self._params.signal_params_loaded()
        self._params_loaded()
        if hasattr(self, 'device'):
            self.to(self.device) 
        self.for_top_level_children(
            lambda n, m: m.signal_params_loaded_recursively())    
    
    def get_profilers(self):
        if hasattr(self, 'use_collector') and self.use_collector == 0:
            from src.codec.coding_tools import ProfilersInterface
            return ProfilersInterface()
        else:
            return self.owner.get_profilers()

    def get_profilers_ctx(self, event, include_name=False):
        from src.codec.coding_tools import ProfilerContext
        e = ''
        if include_name:
            e = self.get_tool_url() + ' '
        e += event
        return ProfilerContext(self.get_profilers, e)

    def init_new_img(self) -> None:
        self._init_new_img()
        self.for_each_child(lambda n,x: x.init_new_img())
        
                
    def export_models_recursively(self, output_dir: str, opset_version: int):
        self.logger.info(f'run export to the folder {output_dir}')
        self.export_models(output_dir, opset_version)
        self.for_top_level_children(lambda _n, m: m.export_models_recursively(os.path.join(output_dir, self.name), opset_version), BaseEngine)        

    # Virtual functions

    def _params_loaded(self) -> None:
        pass

    def export_models(self, output_dir: str, opset_version: int):
        pass
    
    # Hooks
    def _init_new_img(self) -> None:
        pass
