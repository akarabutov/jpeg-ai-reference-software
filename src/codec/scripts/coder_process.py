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

from typing import List
7
from src.codec.coding_tools.coding_engine import CodingEngine
from src.codec.common import Decisions

from .eval import CodecEval


class CoderProcess:
    def __init__(self, coder: CodecEval):
        self.ce = coder.ce if coder is not None else CodingEngine()
            
        self.ce.first_time_exec = True
        self.args_stored = False
        
    def is_model_loaded(self):
        return getattr(self.ce, 'is_model_loaded', False)
    
    def set_model_loaded(self, value):
        setattr(self.ce, 'is_model_loaded', value)
        
    def is_first_time(self):
        return self.ce.first_time_exec 
    
    def set_first_fime(self, value):
        self.ce.first_time_exec = value
        
    def is_args_stored(self):
        return self.args_stored
    
    def set_args_stored(self, value):
        self.args_stored = value
        
    def process(self, cmd_args: List[str]) -> Decisions:
        raise NotImplementedError
        
