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
from .lsbs import LSBSMode
from src.codec.common import Decisions
from src.codec.coding_tools.interfaces import CoderEngine

class LSProcessing(CoderEngine):
    def __init__(self, *args, **kwargs):
        super(LSProcessing, self).__init__(has_enabled_flag=False, *args, **kwargs)
        self.lsbs = LSBSMode()
        self.models_list = list()
        self.models_list.append(self.lsbs)
        
    def _params_loaded(self) -> None:
        ccs_id = self.get_owner_param('ccs_id', 0)
        self.lsbs.enable_flag_name = f"lsbs_enable_flag[{ccs_id}]"
        
    def iter_over_enabled_tools(self):
        for m in self.models_list:
            if m.is_enabled():
                yield m
                
    def iter_rev_over_enabled_tools(self):
        for m in reversed(self.models_list):
            if m.is_enabled():
                yield m        
        
    def analyze(self, decisions: Decisions) -> Decisions:
        return decisions
    
    def pre_processing(self, decisions: Decisions):
        x = decisions.get('y_hat')
        for m in self.iter_over_enabled_tools():
            x = m.pre_processing(x, decisions)
        decisions['y_hat'] = x
    
    def post_processing(self, decisions: Decisions) -> Decisions:
        x = decisions.get('y_hat')
        for m in self.iter_rev_over_enabled_tools():
            x = m.post_processing(x, decisions)
        decisions['y_hat'] = x

        