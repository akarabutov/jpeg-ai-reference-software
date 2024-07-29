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

from src.codec.entropy_coding import HeaderCoder
from typing import List

def decorator_f_header(obj, func):
    def wrapper(*args, **kwargs):
        return func(obj, *args, **kwargs)
    return wrapper

def decorator_o_header(base_obj, obj, func):
    def wrapper(*args, **kwargs):
        return func(base_obj, obj, *args, **kwargs)
    return wrapper

class HeaderBaseFuncs:
        
    @staticmethod
    def enc_f_header(obj: "CoderEngine", hc: HeaderCoder) -> None:
        pass
    
    @staticmethod
    def dec_f_header(obj: "CoderEngine", hc: HeaderCoder) -> None:
        pass
    
    @staticmethod
    def enc_o_header(base_obj: "CoderEngine", obj: "CoderEngine", hc: HeaderCoder) -> None:
        pass
    
    @staticmethod
    def dec_o_header(base_obj: "CoderEngine", obj: "CoderEngine", hc: HeaderCoder) -> None:
        pass        

class HeaderProxy:
    def __init__(self, obj_list: List["CoderEngine"], child_type = None, assign_funcs: bool = True):
        """It connects functions for writing/reading of headers of objects from obj_list or their children with the type child_type if it is presented.
        """
        if child_type is None:
            self.obj_list = obj_list
        else:
            self.obj_list = list()
            for o in obj_list:
                self.obj_list += o.get_children_list(child_type)
        self.funcs = self.obj_list[0].header_base_functions()
        self._process(assign_funcs)

        
    def _process(self, assign_funcs: bool = True) -> None:
        self.obj_list[0].encode_header = decorator_f_header(self.obj_list[0], self.funcs.enc_f_header if assign_funcs else HeaderBaseFuncs.enc_f_header)
        self.obj_list[0].decode_header = decorator_f_header(self.obj_list[0], self.funcs.dec_f_header if assign_funcs else HeaderBaseFuncs.dec_f_header)
        for id in range(1, len(self.obj_list)):
            self.obj_list[id].encode_header = decorator_o_header(self.obj_list[0], self.obj_list[id], self.funcs.enc_o_header if assign_funcs else HeaderBaseFuncs.enc_o_header)
            self.obj_list[id].decode_header = decorator_o_header(self.obj_list[0], self.obj_list[id], self.funcs.dec_o_header if assign_funcs else HeaderBaseFuncs.dec_o_header)
            
            
    def encode_headers(self, hc: HeaderCoder, first_object: bool) -> None:
        if first_object:
            self.funcs.enc_f_header(self.obj_list[0], hc)
        else:
            for id in range(1, len(self.obj_list)):
                self.funcs.enc_o_header(self.obj_list[0], self.obj_list[id], hc)
                
    def decode_headers(self, hc: HeaderCoder, first_object: bool) -> None:
        if first_object:
            self.funcs.dec_f_header(self.obj_list[0], hc)
        else:
            for id in range(1, len(self.obj_list)):
                self.funcs.dec_o_header(self.obj_list[0], self.obj_list[id], hc)
        