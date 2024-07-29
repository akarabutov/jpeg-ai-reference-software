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

import inspect
from src.codec.coding_tools.interfaces import ToolsComposite, ToolsFactory
from src.codec.components import EncoderFactory, DecoderFactory
from src.codec.components.utils import FactoryBase
from .params import ParamsBase, BasicParams
from .model_wrapper import warp_component


def create_wrapped_instance(component_factory: FactoryBase, name: str = "", filter_func=None, *args, **kwargs) -> ToolsComposite:
    tools_dict = dict()
    basic_params = BasicParams()
    for n,v in component_factory.insts.items():
        if filter_func is not None and not filter_func(n):
            continue
        cur_params = ParamsBase()
        init_sign = inspect.getfullargspec(v.__init__)
        start_def_elem = len(init_sign.args) - len(init_sign.defaults)
        for i, an in enumerate(init_sign.args):
            for pn, pa, pk in basic_params.get_params_iter():
                if pn == an:
                    if i >= start_def_elem:
                        pk['default'] = init_sign.defaults[i-start_def_elem]
                    cur_params.add_single_param(pn, *pa, **pk)
                    break
        tools_dict[n] = warp_component(v, cur_params)
    factory = ToolsFactory(tools_dict, name=name)
    ans = ToolsComposite(factory=factory, has_single_tools=True, *args, **kwargs)

    return ans 


def create_dataencoder_instance(name: str = "encoder", filter_func=None, *args, **kwargs) -> ToolsComposite:
    return create_wrapped_instance(EncoderFactory(), name, filter_func=filter_func, *args, **kwargs)

def create_datadecoder_instance(name: str = "decoder", filter_func=None, *args, **kwargs) -> ToolsComposite:
    return create_wrapped_instance(DecoderFactory(), name, filter_func=filter_func, *args, **kwargs)