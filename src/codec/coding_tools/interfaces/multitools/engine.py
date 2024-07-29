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

import numpy as np
import torch

from src.codec.common import Decisions, Image
from src.codec.entropy_coding import HeaderCoder
from ..tool import ToolEngine
##
from .params import MultiToolsParams


class MultiToolsEngine(ToolEngine):
    def __init__(self, *args, **kwargs):
        super(MultiToolsEngine, self).__init__(*args, **kwargs, enabled=True)

        factory = kwargs.get('factory')
        Ntools_max = kwargs.get('Ntools_max', 1)
        self.max_models_count = kwargs.get('max_models_count', None)
        self._params_multi_tool = MultiToolsParams(**kwargs)
        self.tools_common = None

        self.tools = []
        for i in range(Ntools_max):
            self.insert_sub_tool(i, factory.create_instance(has_enabled_flag=False, enabled=True))

        self._active_tool_idx = None

    def insert_sub_tool(self, idx: int, tool: ToolEngine) -> None:
        if self.tools_common is None:
            from src.codec.coding_tools.interfaces import ParamsCommonObj
            self.tools_common = ParamsCommonObj([tool])
        self.tools.append(tool)
        setattr(self, f'tools_{idx}', tool)
        self.tools_common.target_objs = self.tools   

    def remove_sub_tool(self, idx: int) -> None:
        del self.tools[idx]
        delattr(self, f'tools_{idx}')
        self.tools_common.target_objs = self.tools

    @property
    def active_tool_idx(self) -> int:
        return self._active_tool_idx if self._active_tool_idx is not None else int(
            np.floor(self.get_target_bpp_idx() / 4 * len(self.tools)))
        
    def get_active_tool_idx(self) -> int:
        return self.active_tool_idx

    @active_tool_idx.setter
    def active_tool_idx(self, val: int) -> None:
        self._active_tool_idx = val

    def get_active_tool(self):
        return self.tools[self.active_tool_idx]

    def compress(self, img: Image, *args, **kwargs) -> Decisions:
        ans = self.get_active_tool().compress(img, *args, **kwargs)
        ans['tool_id'] = self.active_tool_idx
        return ans

    def decompress(self, decisions, *args, **kwargs):
        return self.get_active_tool().decompress(decisions, *args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.get_active_tool().encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        ans = self.get_active_tool().decode(*args, **kwargs)
        # Store tool id
        ans['tool_id'] = self.active_tool_idx
        return ans

    def load_model(self, downloader):
        cp_model_name = self.get_model_name()
        cp_files = self.get_ckpt_names()

        for i, cp_fn in enumerate(cp_files):
            if self.tools[i].is_enabled():
                full_cp_fn = downloader.get_file_path(cp_model_name, cp_fn)
                cp = torch.load(full_cp_fn, map_location=self.tools[i].device)
                self.logger.info(f"=> loading checkpoint '{cp_fn}' for tool {i}")
                self.tools[i].load_state_dict(cp, strict=True)

    def encode_header(self, ec: HeaderCoder):
        max_models_count = len(self.tools) if self.max_models_count is None else self.max_models_count
        ec.encode(self.active_tool_idx, max_models_count-1, name='model_id')
        self.__disable_inactive_tools()

    def __disable_inactive_tools(self):
        act_idx = self.active_tool_idx
        for i, m in enumerate(self.tools):
            m.set_enable(i == act_idx)

    def decode_header(self, ec: HeaderCoder):
        max_models_count = len(self.tools) if self.max_models_count is None else self.max_models_count
        self.active_tool_idx = int(
            ec.decode([1], max_models_count-1, name='model_id').item())
        self.__disable_inactive_tools()

    def _params_loaded(self):
        if self.Ntools is None:
            self.Ntools = len(self.get_ckpt_names())
        Nmax = len(self.tools)
        for i in range(Nmax):
            # Disable loading tools, because it should be buisiness of this module
            self.tools[i].ckpt_model_name = ''
        for i in range(Nmax - 1, self.Ntools - 1, -1):
            self.remove_sub_tool(i)
