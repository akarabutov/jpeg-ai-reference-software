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

##
import torch
import torch.nn as nn

from .ec_dump import ECDump
from .header_module import HeaderCoder


class ECModule(nn.Module):
    check_range = True

    def __init__(self, bs: "BitstreamStructure", verbose:bool=False, dump_tool:ECDump=None, *args, **kwargs):
        super(ECModule, self).__init__()
        from src.codec.coding_tools.profiler import ProfilersInterface
        self.__verbose = verbose
        self.bs = bs
        self.profilers = kwargs.get('profilers', ProfilersInterface())
        self._stream_base_comp = 0
        self._stream_part = "pic_header"
        self._substream_name_hr = None
        self._region_idx = 0
        self._dump_tool = dump_tool
        self._update_wrappers()
        
    @property
    def stream_part(self):
        return self._stream_part

    @property
    def region_idx(self):
        return self._region_idx

    @stream_part.setter
    def stream_part(self, value: int):
        if self._stream_part != value:
            self._stream_part = value
            self._update_wrappers()

    @region_idx.setter
    def region_idx(self, value: int):
        if self._region_idx != value:
            self._region_idx = value
            self._update_wrappers()

    @property
    def stream_base_comp(self):
        return self._stream_base_comp


    @stream_base_comp.setter
    def stream_base_comp(self, value: int):
        if self._stream_base_comp != value:
            self._stream_base_comp = value
            self._update_wrappers()
    
    def _update_wrappers(self):
        from src.codec.bitstream_structure import SubstreamLayouts
        substream_layout = SubstreamLayouts.get_substreamtype_by_name(self._stream_base_comp==0, self._stream_part)
        self.ec_lib = self.bs.get_ec(substream_layout, region_idx=self._region_idx)
        self._substream_name_hr = substream_layout.human_readable_name
        # wrappers for supported ProbModels
        prob_wrappers = self.ec_lib.get_wrapper_probs()
        self.custom = prob_wrappers.get('Custom', None)
        self.bypass = prob_wrappers.get('Bypass', None)
        self.sgt = prob_wrappers.get('Sgt', None)
        

    def get_eclib_name(self):
        return self.ec_lib.get_name()

    def get_total_bits(self):
        return self.ec_lib.get_total_bits()

    def reset_label_attrs(self):
        return self.ec_lib.reset_label_attrs()
    
    def get_header_codec(self) -> HeaderCoder:
        return HeaderCoder(self)
    
    def get_substream_size(self) -> int:
        return self.ec_lib.get_mem_size()
    
    def _mark_ae(self):
        self.ec_lib.ae_used = True

    # ##################################################################################################################
    #  Wrappers for ProbModels
    # ##################################################################################################################
    ''' Wrapper for SgtProbModel '''

    def encode_sgt(self,
                   x: torch.Tensor,
                   sigma: torch.Tensor,
                   masks: torch.Tensor,
                   name: str = None,
                   entropy_prob_model=None):
        """Encode signal by Table based Single Gaussian model

        Args:
            x (torch.Tensor): input signal.
            sigma (torch.Tensor): .
            name (str, optional): Label for data. Defaults to None.
            entropy_prob_model: is an prob. model
        """
        if self.__verbose:
            print(f'Encode SGT for {name} to stream {self._stream_part}')
        self.profilers.start(f'Encode SGT {name}')
        self.sgt.encode(x, sigma, masks, name, entropy_model=entropy_prob_model)
        self.profilers.finish(f'Encode SGT {name}')
        self._mark_ae()

    def decode_sgt(self, sigma: torch.Tensor, masks: torch.Tensor, name: str = None, entropy_prob_model=None):
        """Decode signal by Table based Single Gaussian model

        Args:
            sigma (torch.Tensor): .
            name (str, optional): Label for data. Defaults to None.
            entropy_prob_model: is an prob. model

        """
        if self.__verbose:
            print(f'Decode SGT for {name} from stream {self._stream_part}')
        self.profilers.start(f'Decode SGT {name}')
        x = self.sgt.decode(sigma, masks, name, entropy_model=entropy_prob_model)
        self.profilers.finish(f'Decode SGT {name}')
        if self._dump_tool is not None:
            self._dump_tool.store_sgt(self._substream_name_hr, self._region_idx, name, list(x.shape))
        self._mark_ae()
        return x

    ''' Wrapper for CustomProbModel '''

    def encode_custom(self, x, model, max_symbol_value=512, mean=None, name=None):
        if self.__verbose:
            print(f'Encode custom for {name} to stream {self._stream_part}')
        self.profilers.start(f'Encode Custom {name}')
        self.custom.encode(x, model, max_symbol_value, mean, name)
        self.profilers.finish(f'Encode Custom {name}')
        self._mark_ae()

    def decode_custom(self, shape, model, max_symbol_value=512, mean=None, name=None):
        if self.__verbose:
            print(f'Decode custom for {name} from stream {self._stream_part}')
        self.profilers.start(f'Decode Custom {name}')
        x = self.custom.decode(shape, model, max_symbol_value, mean, name)
        self.profilers.finish(f'Decode Custom {name}')
        if self._dump_tool is not None:
            self._dump_tool.store_factorized(self._substream_name_hr, self._region_idx, name, list(x.shape))
        self._mark_ae()
        return x

    ''' Wrapper for BypassProbModel '''

    def encode_bypass(self, x: torch.Tensor, max_symbol_value=1, name=None):
        if self.__verbose:
            print(f'Encode bypass for {name} = {x.flatten()} to stream {self._stream_part}')
        self.profilers.start(f'Encode bypass {name}')
        self.bypass.encode(x, max_symbol_value, name)
        self.profilers.finish(f'Encode bypass {name}')
        self._mark_ae()

    def decode_bypass(self, shape, max_symbol_value=1, device=torch.device('cpu'), name=None) -> torch.Tensor:
        self.profilers.start(f'Decode bypass {name}')
        x = self.bypass.decode(shape, max_symbol_value, device, name)
        self.profilers.finish(f'Decode bypass {name}')
        self._mark_ae()
        if self._dump_tool is not None:
            self._dump_tool.store_bypass(self._substream_name_hr, self._region_idx, name, x)
        if self.__verbose:
            print(f'Decode bypass for {name} = {x.flatten()} from stream {self._stream_part}')
        return x

class ECModuleLH(ECModule):
    def __init__(self):
        from src.codec.bitstream_structure import BitstreamStructure
        from .lib_wrappers import ECLibLH
        bs = BitstreamStructure(ECLibLH, coder_direction=0)
        super(ECModuleLH, self).__init__(bs)
    
    def _update_wrappers(self):
        from .lib_wrappers import ECLibLH
        
        # wrappers for supported ProbModels
        ec_lib = getattr(self, 'ec_lib', ECLibLH())
        prob_wrappers = ec_lib.get_wrapper_probs()
        self.custom = prob_wrappers.get('Custom', None)
        self.bypass = prob_wrappers.get('Bypass', None)
        self.sgt = prob_wrappers.get('Sgt', None)    
        self.ec_lib = ec_lib

def create_lh_ecmodule():
    return ECModuleLH()

    