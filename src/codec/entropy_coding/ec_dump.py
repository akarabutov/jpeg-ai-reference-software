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

import torch
from prettytable import PrettyTable
from typing import Any, List, Tuple, Union
from collections import OrderedDict
import commentjson

class ECDump:
    """Class for dumping of decoded flags
    """
    def __init__(self):
        self.data = OrderedDict()
        self.fields = ['Parameters', 'Flags', 'Gaussian data', 'Factorized data']
        
    def _create_stream(self, stream_name: str) -> None:
        if stream_name not in self.data:
            self.data[stream_name] = OrderedDict()        
        
    def _create_region(self, stream_name: str, region_id: Union[int, str]) -> str:
        self._create_stream(stream_name)
        
        region_name = self._gen_region_name(region_id) if isinstance(region_id, int) else region_id
        if region_name not in self.data[stream_name]:
            self.data[stream_name][region_name] = OrderedDict()
            for f_name in self.fields:
                self.data[stream_name][region_name][f_name] = OrderedDict()
            
        return region_name
            
    @staticmethod
    def _gen_region_name(region_idx: int) -> str:
        return f"region_{region_idx}"
    
    def _check_region_value(self, stream_name: str, region_name: str) -> None:
        fields_has_data_num = 0
        for data_name in self.fields:
            if data_name != 'Parameters':
                fields_has_data_num += 1 if len(self.data[stream_name][region_name][data_name]) > 0 else 0
        
        assert fields_has_data_num < 2
        
    def add_substream_params(self, stream_name: str, region_idx: int, params: OrderedDict):
        region_name = self._create_region(stream_name, region_idx)
        self.data[stream_name][region_name]['Parameters'].update(params)
        
    def store_bypass(self, stream_name: str, region_id: int, flag_name: str, flag_value: Any) -> None:
        region_name = self._create_region(stream_name, region_id)
        if isinstance(flag_value, torch.Tensor):
            if flag_value.numel() == 1:
                flag_value = flag_value.item()
            else:
                flag_value = flag_value.view([-1]).tolist()
        self.data[stream_name][region_name]['Flags'][flag_name] = flag_value
        self._check_region_value(stream_name, region_name)
        
    def store_sgt(self, stream_name: str, region_id: int, field_name: str, field_size: List[int]) -> None:
        region_name = self._create_region(stream_name, region_id)
        self.data[stream_name][region_name]['Gaussian data'][field_name] = field_size
        self._check_region_value(stream_name, region_name)
        
    def store_factorized(self, stream_name: str, region_id: int, field_name: str, field_size: List[int]) -> None:
        region_name = self._create_region(stream_name, region_id)
        self.data[stream_name][region_name]['Factorized data'][field_name] = field_size
        self._check_region_value(stream_name, region_name)
        
    def store(self, filename) -> None:
        with open(filename, 'w') as f:
            commentjson.dump(self.data, f, indent='\t')
            
    def load(self, filename) -> None:
        with open(filename, 'r') as f:
            self.data = commentjson.load(f)
            
    @staticmethod
    def print_values(values: List[OrderedDict]) -> None:
        table = PrettyTable()
        table.field_names = ["Name", "Value"]
        for (n,v) in values.items():
            #print(f'{indent}{n} = {v}')
            table.add_row([n,v])
        print(table)


    def print(self) -> None:
        for section_name, section_value in self.data.items():
            print(f"Substream '{section_name}':")
            for region_name, region_value in section_value.items():
                print(f"= {region_name}:")
                for fn, fv in region_value.items():
                    if isinstance(fv, OrderedDict) and len(fv):
                        print(f"== {fn}:")
                        self.print_values(fv)