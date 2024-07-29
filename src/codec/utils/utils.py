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

from typing import Dict, List, Union

from src.codec.common import ArgParserDecorator
from src.codec.common.utils import update_dict_recursively

from ..coding_tools.interfaces import BaseEngine

from argparse import _AppendAction

def _copy_items(items):
    if items is None:
        return []
    # The copy module is used only in the 'append' and 'append_const'
    # actions, and it is needed only when the default value isn't a list.
    # Delay its import for speeding up the common case.
    if type(items) is list:
        return items[:]
    import copy
    return copy.copy(items)

class ExtendAction(_AppendAction):
    # Borrowed from python 3.11
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, None)
        items = _copy_items(items)
        items.extend(values)
        setattr(namespace, self.dest, items)

def broadcast_params(base_list: List[BaseEngine], configs: Dict) -> None:
    for b in base_list:
        b.load_params2attrs_recursively(**configs)
        
def cmd_params_loading(parser: ArgParserDecorator,
                       base: Union[BaseEngine, List[BaseEngine]],
                       cfg_path: Union[List, str] = None,
                       params_preprocess=None,
                       cmd_args: List = None) -> Dict:

    base_list = base
    if isinstance(base, BaseEngine):
        base_list = [base]

    for b in base_list:
        b.set_defval2attrs_recursively()
    if cfg_path is not None:
        full_cfg_list = parser.get_cfgs(cfg_path)
        for cfg in full_cfg_list:
            broadcast_params(base_list, parser.load_params_from_cfg_file(cfg))
    broadcast_params(base_list, parser.load_params_from_cmd_line(args=cmd_args))

    ans = dict()

    for b in base_list:
        ans = update_dict_recursively(ans, b.store_attrs2dict_recursively())
    
    ans = params_preprocess(ans)    
    broadcast_params(base_list, ans)
    for b in base_list:
        b.signal_params_loaded_recursively()

    return ans
