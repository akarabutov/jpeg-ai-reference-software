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

from typing import List

from src.codec.common import Decisions, Image
from src.codec.common.utils import pop_param_from_dict
from src.codec.entropy_coding import ECModule, HeaderCoder
from src.codec.coding_tools.interfaces import CoderEngine

from ..model import ModelMgmtParams
from ..rdo import RdoPreProcInterface
from ..tool import ToolEngine
##
from .params import CompositeParams


class ToolsComposite(ToolEngine):
    def __init__(self, *args, **kwargs):
        kwargs, factory = pop_param_from_dict(kwargs, 'factory', None)                      # Factory object
        kwargs, has_single_tools = pop_param_from_dict(kwargs, 'has_single_tools', False)   # The composite has only one tool for inference
        kwargs, tools_set_once = pop_param_from_dict(kwargs, 'tools_set_once', None)        # Parameter tool/tools could be set once 
        kwargs, remove_unused_tools = pop_param_from_dict(kwargs, 'remove_unused_tools', True)        # Remove unused tools after setting of tool(s)
        super(ToolsComposite, self).__init__(has_enabled_flag=False, *args, **kwargs)
        self._params_composite = CompositeParams(has_single_tools,
                                                 tools_name=factory.get_tools_name())
        
        # Internal parameters
        self._has_single_tools = has_single_tools
        self._remove_usused_tools = remove_unused_tools
        self._tools_set = False if tools_set_once else None

        # Dictionary with tools, which were additionally added
        self.params.remove_param_inst(ModelMgmtParams)
        self._all_tools_name = factory.get_tools_name()
        self._tools_order = factory.get_tools_name()
        self.__dict__['preproc_tools'] = list()
        self.__dict__['postproc_tools'] = list()
        kwargs, _ = pop_param_from_dict(kwargs, 'has_enabled_flag', False)
        for tn in self._all_tools_name:
            inst = factory.create_instance(tn, has_enabled_flag=not has_single_tools, **kwargs)
            setattr(self, tn, inst)
            
    def set_remove_unused_tools(self, value: bool) -> None:
        self._remove_usused_tools = value
        
    def get_tool(self, idx = 0) -> ToolEngine:
        name = self._tools_order[idx] if idx < len(self._tools_order) else self._tools_order[-1]
        return getattr(self, name)

    def _set_tools_order(self, tools_list: List[str]) -> None:
        self._tools_order = tools_list

    def preserve_only_tools(self, tools_list: List[str]) -> None:
        self._set_tools_order(tools_list)        
        if not self._remove_usused_tools:
            return
        for tn in self._all_tools_name:
            if tn not in tools_list and hasattr(self, tn):
                delattr(self, tn)
        
    def get_tools_list(self) -> List[str]:
        return self._all_tools_name

    def is_enabled(self):
        ans = False
        for tv in self.iter_over_tools():
            ans = ans or tv.is_enabled()
        return ans

    def add_preproc_tool(self, name: str, tool: RdoPreProcInterface) -> None:
        self.__dict__['preproc_tools'].append(tool)
        self.add_sub_tool(name, tool)

    def add_postproc_tool(self, name: str, tool: RdoPreProcInterface) -> None:
        self.__dict__['postproc_tools'].append(tool)
        self.add_sub_tool(name, tool)

    def compress(self, image: Image, *args, **kwargs) -> Decisions:
        ans = {}
        current_image = image.clone()
        tool_processed = False
        for i, (tn, tv) in enumerate(self.iter_over_naming_tools(only_enabled_tools=True)):
            if tv.is_enabled():
                is_last = i == (len(self._tools_order) - 1)
                for ppt in self.__dict__['preproc_tools']:
                    if ppt.enabled:
                        ppt.process(tv, current_image)
                    else:
                        ppt.process_default(tv)
                img_temp = current_image.clone()
                img_temp.input_file = current_image.input_file
                ans[tn] = tv.compress(img_temp, *args, **kwargs)
                for ppt in self.__dict__['postproc_tools']:
                    if ppt.enabled:
                        img_temp = current_image.clone()
                        img_temp.input_file = current_image.input_file
                        ans[tn] = ppt.process(tv, img_temp, ans[tn])
                tool_processed = True
                if not is_last:
                    current_image = tv.decompress(ans[tn], *args, **kwargs)
        return ans if tool_processed else image

    # The second stage on decoder-side: decompression latent representation to image
    def decompress(self, decisions: Decisions, ref_img=None, *args, **kwargs) -> Image:
        ans = None
        cur_ref_img = ref_img
        tool_processed = False
        for tn, tv in self.iter_over_naming_tools(only_enabled_tools=True):
            # Store only the last image
            ans = tv.decompress(decisions.get(tn, {}), ref_img=cur_ref_img, *args, **kwargs)
            tool_processed = True
            cur_ref_img = ans
        return ans if tool_processed else decisions

    # The second stage on encoder-side: encoding latent information to bitstream
    def encode(self, ec: ECModule, decisions: Decisions) -> None:
        for tn, tv in self.iter_over_naming_tools(only_enabled_tools=True):
            tv.encode(ec, decisions.get(tn, {}))

    # The first stage on decoder-side: decoding latent information from bitstream
    def decode(self, ec: ECModule) -> Decisions:
        ans = Decisions()
        for tn, tv in self.iter_over_naming_tools(only_enabled_tools=True):
            ans[tn] = tv.decode(ec)
        return ans
    
    def forward(self, *args, **kwargs):
        ans = None
        for t in self.iter_over_tools(True):
            ans = t(*args, **kwargs)
        return ans

    def iter_over_tools(self, only_enabled_tools=False):
        for _, tv in self.iter_over_naming_tools(only_enabled_tools=only_enabled_tools):
            yield tv

    def iter_over_naming_tools(self, only_enabled_tools=False):
        for tn in self._tools_order:
            tv = getattr(self, tn)
            if not only_enabled_tools or tv.is_enabled():
                yield tn, tv

    def iter_rev_over_naming_tools(self, only_enabled_tools=False):
        for tn in reversed(self._tools_order):
            tv = getattr(self, tn)
            if not only_enabled_tools or tv.is_enabled():
                yield tn, tv

    def _params_loaded(self):
        if self._tools_set is None or not self._tools_set:
            super()._params_loaded()
            if self._has_single_tools:
                tools_list = list()
                if self.tool is not None:
                    tools_list.append(self.tool)
            else:
                tools_list = self.tools
                if '' in tools_list:
                    del tools_list[tools_list.index('')]
            self.preserve_only_tools(tools_list)
        if self._tools_set is not None:
            self._tools_set = True

    # Recursive functions
    def encode_header_recursively(self, ec: HeaderCoder) -> None:
        with self.set_ec_context(ec, self._stream_header_part):
            if self.enabled:
                for m in self.iter_over_tools(False):
                    if isinstance(m, CoderEngine):
                        m.encode_header_recursively(ec)

    def decode_header_recursively(self, ec: HeaderCoder) -> None:
        with self.set_ec_context(ec, self._stream_header_part):        
            if self.enabled:
                for m in self.iter_over_tools(False):
                    if isinstance(m, CoderEngine):
                        m.decode_header_recursively(ec)
            else:
                self.init_decoder()
