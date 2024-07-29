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

from typing import List, Union

from ..base import BaseModule


class AttrsProxy(object):
    def __init__(self,
                 attr_name_list: List[str],
                 dest_objs: Union[BaseModule, List[BaseModule]],
                 source_obj: BaseModule = None):
        self.__source_obj = source_obj
        self.attr_name_list = attr_name_list
        if isinstance(dest_objs, BaseModule):
            dest_objs = [dest_objs]
        self.dest_objs = dest_objs

    @property
    def source_obj(self) -> BaseModule:
        return self.__source_obj
    
    @property
    def dest_objs(self) -> List[BaseModule]:
        return self.__dest_objs

    @source_obj.setter
    def source_obj(self, source_obj: BaseModule):
        self.__source_obj = source_obj
        
    @dest_objs.setter
    def dest_objs(self, value: List[BaseModule]):
        self.__dest_objs = value

    def process(self):
        for dest_obj in self.dest_objs:
            self.__source_obj.copy_attrs_value(dest_obj, self.attr_name_list)
