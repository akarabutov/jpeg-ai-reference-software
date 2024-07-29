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

from typing import List, Dict
from xmlrpc.client import Boolean

import numpy
import torch

from src.codec.common.utils import (copy_dict_recursively, update_dict_recursively)


class Decisions(dict):
    def add_postfix_to_keys_(self, postfix: str, exclude_list: List[str] = list()) -> 'Decisions':
        convert_dict = {}
        for k, v in self.items():
            if k not in exclude_list:
                convert_dict[k] = f'{k}{postfix}'

        for k_old, k_new in convert_dict.items():
            self[k_new] = self[k_old]
            del self[k_old]

        return self
    
    def to_f32_(self):
        self.apply_(lambda x: x.to(torch.float32))

    def detach(self):
        self.apply_(lambda x: x.detach())
        
    def permute(self, *args):
        self.apply_(lambda x: x.permute(*args))
        
    def preserve_only(self, *args):
        self.__preserve_only(self, *args)
        
    def apply_(self, action):
        self = self.__apply(self, action)
        
    def __apply(self, orig_dict: Dict[str, torch.Tensor], action) -> Dict[str, torch.Tensor]:
        for key, val in orig_dict.items():
            if isinstance(val, dict):
                tmp = self.__apply(val, action)
                orig_dict[key] = tmp
            else:
                if isinstance(orig_dict[key], torch.Tensor):                       
                    orig_dict[key] = action(orig_dict[key])
        return orig_dict
    
    def __preserve_only(self, orig_dict: Dict[str, torch.Tensor], *args) -> Dict[str, torch.Tensor]:
        keys_for_rem = list()
        for key, val in orig_dict.items():
            if isinstance(val, dict):
                tmp = self.__preserve_only(val, *args)
                orig_dict[key] = tmp
            else:
                if key not in args:         
                    keys_for_rem.append(key)              
        for k in keys_for_rem:
            del orig_dict[k]
        return orig_dict    

    def select_keys_by_postfix(self, postfix: str, preserve_postfix=True) -> 'Decisions':
        ans = Decisions()
        for k, v in self.items():
            if k.endswith(postfix):
                if not preserve_postfix:
                    k = k[:-len(postfix)]
                ans[k] = v
        return ans

    def remove_keys_(self, keys_list: List[str]) -> 'Decisions':
        for k in keys_list:
            if k in self:
                del self[k]
        return self

    def has_keys_with_postfix(self, postfix: str) -> Boolean:
        ans = False
        for k in self.keys():
            if k.endswith(postfix):
                ans = True
                break
        return ans

    def is_empty(self) -> Boolean:
        return len(self.keys()) == 0

    def clone(self) -> 'Decisions':
        ans = Decisions()
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                ans[k] = v.detach().clone()
            elif numpy.isscalar(v):
                ans[k] = v
            elif isinstance(v, Decisions):
                ans[k] = v.clone()
            else:
                if v is None:
                    ans[k] = None
                else:
                    ans[k] = copy_dict_recursively(v)
        return ans

    def update(self, decisions: 'Decisions') -> None:
        self = update_dict_recursively(self, decisions)
