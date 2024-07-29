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

from typing import Dict, List

##
from .engine import ToolEngine


class ToolsFactory(object):
    def __init__(self, tools_dict: Dict[str, ToolEngine] = None, default_tools: List[str] = None, name: str = ''):
        self.tools_dict = tools_dict
        self.default_tools = default_tools
        self.name = name

    def create_instance(self, tool_name: str, *args, **kwargs) -> ToolEngine:
        ans = self.tools_dict.get(tool_name, None)
        if ans is None:
            raise ValueError(f'Wrong {self.name} name "{tool_name}"')

        return ans(*args, **kwargs)

    def get_tools_name(self) -> List[str]:
        """Get list of tools

        Returns:
            List[str]: list of tools
        """
        return list(self.tools_dict.keys())

    def get_default_tools(self) -> List[str]:
        """Get list of default tools

        Returns:
            List[str]: list of default tools
        """
        if self.default_tools is None:
            return self.get_tools_name()
        elif isinstance(self.default_tools, str):
            return [self.default_tools]
        else:
            return self.default_tools
