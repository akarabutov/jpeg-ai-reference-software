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

from typing import Union

from src.codec.common import Decisions, Image

from ..base import BaseEngine
from ..multitools import MultiToolsEngine
from ..tool import ToolEngine
from .params import RdoParams


class RdoPreProcInterface(BaseEngine):
    def __init__(self, *args, **kwargs):
        super(RdoPreProcInterface, self).__init__(*args, **kwargs)
        self.__params_rdo = RdoParams()

    def process(self, tool: ToolEngine, ori_img: Image) -> None:
        raise NotImplementedError
    
    def process_default(self, tool: ToolEngine) -> None:
        """Call this function when the tool is disabled
        """
        raise NotImplementedError


class RdoPostProcInterface(BaseEngine):
    def __init__(self, *args, **kwargs):
        super(RdoPostProcInterface, self).__init__(*args, **kwargs)
        self.supported_tools_types = list()
        self.__params_rdo = RdoParams()

    def check_tool(self, tool: ToolEngine) -> bool:
        """Check that tool supported by RDO

        Returns:
            bool: is tool supported or not
        """
        ans = False
        for t in self.supported_tools_types:
            if isinstance(tool, t):
                ans = True
        return ans

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

    def _process(self, tool: ToolEngine, ori_img: Image, decisions: Decisions) -> Decisions:
        # Based on original image and generated decisions generate tuned decisions for the tool
        raise NotImplementedError

    def process(self, tool: Union[ToolEngine, MultiToolsEngine], ori_img: Image,
                decisions: Decisions) -> Decisions:
        if isinstance(tool, MultiToolsEngine):
            tool = tool.get_active_tool()
        return self._process(tool, ori_img, decisions) if self.check_tool(tool) else decisions
