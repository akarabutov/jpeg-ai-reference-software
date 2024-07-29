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

import torch.nn as nn


class BaseModule(nn.Module):
    """BaseModule class, which supports pyramidal hierarchical structure:
        - Call from root to leaves and back
        - Storing parameters information

    """

    # Public functions
    def __init__(self, *args, **kwargs):
        super(BaseModule, self).__init__()
        self.__name = str()
        self.__owner = None

    def add_sub_tool(self, name: str, tool: 'BaseModule'):
        setattr(self, name, tool)

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str) -> None:
        self.__name = name

    # Access to """ owner
    @property
    def owner(self) -> 'BaseModule':
        assert self.__dict__['__owner'] is not None
        return self.__dict__['__owner']

    #@owner.setter
    def set_owner(self, parent: 'BaseModule'):
        self.__dict__['__owner'] = parent

    # Service functions
    def has_owner(self) -> bool:
        return '__owner' in self.__dict__ and self.__dict__['__owner'] is not None

    def get_tool_url(self) -> str:
        """
        Get URL of tool, which includes name of all owners in a tree
        """
        return self.owner.get_tool_url() + '.' + self.name if self.has_owner() else 'base'

    # Function for internal usage
    def __setattr__(self, name: str, value) -> None:
        if isinstance(value, BaseModule) and name != 'owner':
            value.name = name
            value.__dict__['__owner'] = self
            #value.owner = self
        super(BaseModule, self).__setattr__(name, value)

    def __getattr__(self, name: str):
        if name.startswith('get_'):
            # Pass methods started with 'get_' from owner
            try:
                ans = super().__getattr__(name)
            except AttributeError:
                ans = getattr(self.owner, name)
        else:
            ans = super().__getattr__(name)
        return ans
    

    def get_children_list(self, flt, recursive: bool = True) -> List['BaseModule']:
        ans = list()
        for m in self.children():
            if isinstance(m, flt):
                ans.append(m)
            if recursive and isinstance(m, BaseModule):
                ans += m.get_children_list(flt)
        return ans
            

    def copy_attrs_value(self, other: 'BaseModule', attrs_list: List[str]) -> None:
        for n in attrs_list:
            if hasattr(self, n):
                setattr(other, n, getattr(self, n))

    def set_attrs_value(self, attrs: dict) -> None:
        for n, v in attrs.items():
            setattr(self, n, v)

    def generate_attrs_dict(self,
                            params_list: List[str],
                            substr: str,
                            removal_substrs: List[str],
                            with_substr: bool = True) -> Dict:
        ans = {}
        for p in params_list:
            if ((substr in p) and with_substr) or ((substr not in p) and not with_substr):
                # Remove sub strings
                p_out = p
                for rs in removal_substrs:
                    p_out = p_out.replace(rs, '')
                ans[p_out] = getattr(self, p)
        return ans

    def get_owner_param(self, param_name: str, def_val=None):
        if hasattr(self, param_name):
            return getattr(self, param_name, def_val)
        else:
            return self.owner.get_owner_param(param_name, def_val)

    def named_children_flt(self, subclass_name):
        for name, m in self.named_children():
            if isinstance(m, subclass_name):
                yield name, m

    def for_each_child(self, fn, flt=None):
        for name, m in self.named_children_flt(flt if flt is not None else BaseModule):
            fn(name, m)
            m.for_each_child(fn, flt)

    def for_top_level_children(self, fn, flt=None):
        for name, m in self.named_children_flt(flt if flt is not None else BaseModule):
            fn(name, m)

    def get_object_by_url(self, url: str) -> "BaseModule":
        cur_url = self.get_tool_url()
        if cur_url == url:
            return self
        if url.startswith(cur_url):
            rest_url = url[len(cur_url)+1:]
            rest_url_list = rest_url.split(".")
            if hasattr(self, rest_url_list[0]) and isinstance(getattr(self, rest_url_list[0]), BaseModule):
                obj =  getattr(self, rest_url_list[0])
                return obj if len(rest_url_list)==1 else obj.get_object_by_url(url)
            else:
                return None
        else:
            return self.owner.get_object_by_url(url)
            