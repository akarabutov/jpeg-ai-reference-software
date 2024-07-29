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
import numpy as np
from typing import List
from src.codec.common import Log2LinConvertion, Decisions
from src.codec.coding_tools.interfaces import CoderEngine, HeaderBaseFuncs
from .gain_unit import GainUnit
# from .quality_map import QualityMap
from .params import QuantizerParams
from .rvs import ResVarScale
from src.codec.entropy_coding import HeaderCoder, ECModule
from .base import QuantizationInterface

class QuantizerHeaderBaseFuncs:
    
    @staticmethod
    def enc_flag(obj: "Quantizer", hc: HeaderCoder, idx: int) -> None:
        # betaDisplacementLog  clipping -- start
        if obj.beta_displacement_log < obj.beta_displacement_log_low_bound:
            obj.logger.debug(f"The betaDisplacementLog value {obj.beta_displacement_log} will be clipped to {obj.beta_displacement_log_low_bound}")
        elif obj.beta_displacement_log > obj.beta_displacement_log_high_bound:
            obj.logger.debug(f"The betaDisplacementLog value {obj.beta_displacement_log} will be clipped to {obj.beta_displacement_log_high_bound}")
        else:
            obj.logger.debug(f"The betaDisplacementLog value {obj.beta_displacement_log} will not be clipped")
            
        obj.beta_displacement_log = np.clip(obj.beta_displacement_log, obj.beta_displacement_log_low_bound, obj.beta_displacement_log_high_bound)
        # betaDisplacementLog  clipping -- end    
        
        hc.encode(obj.beta_displacement_log + 2**(obj.beta_displacement_log_bitdepth-1), bits_count=12, name=f'beta_displacement_log_plus_2048[{idx}]')
        obj.logger.debug(f"beta_displacement_log[{idx}] = {obj.beta_displacement_log}")
        
    @staticmethod
    def enc_f_header(obj: "Quantizer", hc: HeaderCoder) -> None:
        QuantizerHeaderBaseFuncs.enc_flag(obj, hc, 0)
    
    @staticmethod
    def enc_o_header(base_obj: "Quantizer", obj: "Quantizer", hc: HeaderCoder) -> None:
        use_different_betas_for_Y_and_UV = 0 if base_obj.beta_displacement_log == obj.beta_displacement_log else 1
        hc.encode(use_different_betas_for_Y_and_UV, 1, name="independent_beta_uv")
        if use_different_betas_for_Y_and_UV:
            QuantizerHeaderBaseFuncs.enc_flag(obj, hc, 1)
    
    @staticmethod
    def dec_flag(obj: "Quantizer", hc: HeaderCoder, idx: int) -> None:
        obj.beta_displacement_log = hc.decode([1], bits_count=12, name=f'beta_displacement_log_plus_2048[{idx}]').item() - 2**(obj.beta_displacement_log_bitdepth-1)
        obj.logger.debug(f"beta_displacement_log[{idx}] = {obj.beta_displacement_log}")        

    @staticmethod
    def dec_f_header(obj: "Quantizer", hc: HeaderCoder) -> None:
        QuantizerHeaderBaseFuncs.dec_flag(obj, hc, 0)

    @staticmethod
    def dec_o_header(base_obj: "Quantizer", obj: "Quantizer", hc: HeaderCoder) -> None:
        use_different_betas_for_Y_and_UV = hc.decode(1, 1, name="independent_beta_uv")
        if use_different_betas_for_Y_and_UV:
            QuantizerHeaderBaseFuncs.dec_flag(obj, hc, 1)
        else:
            obj.beta_displacement_log = base_obj.beta_displacement_log

class Quantizer(CoderEngine):
    def __init__(self, 
                 chs: int, 
                 log_k: float = (np.log(54.82)-np.log(0.11))/31,
                 *args, **kwargs):
        super(Quantizer, self).__init__(use_coding_headers=True, has_enabled_flag=False, *args, **kwargs)
        self.log_k = log_k
        self.rvs = ResVarScale(chs, stream_header_part="pic_header")
        self.gain_unit = GainUnit(chs, log_k=log_k, stream_header_part="tool_header")
        #self.qual_map = QualityMap(stream_header_part="tool_header")
        self.__beta_displacement_log: int = 0
               
        self._params_precisions = QuantizerParams()
    
    @staticmethod
    def header_base_functions() -> HeaderBaseFuncs:
        """Virtual function for returning of a set of base functions for reading/writing headers
        """
        return QuantizerHeaderBaseFuncs        
        
    @property
    def beta_displacement(self) -> float:
        return Log2LinConvertion.difflog2lin(torch.ones(1,1,1,1)*self.beta_displacement_log, self.sigma_precision, self.log_k)
    
    @property
    def beta_displacement_log(self) -> int:
        return self.__beta_displacement_log    
    
    @beta_displacement_log.setter
    def beta_displacement_log(self, value: int) -> None:
        self.__beta_displacement_log = value
        for m in self.iter_over_enabled_tools(only_quant_int=True):
            m._beta_displacement_log_updated(value)
        
    @property
    def unscaled_sigma_precision(self) -> int:
        return self.sigma_precision
    
    @property
    def scaled_sigma_precision(self) -> int:
        return self.scaler_precision + self.sigma_precision
        
    def _params_loaded(self) -> None:
        self.__dict__['qual_map'] = self.get_owner_param('qual_map')
        self.models_list = list()
        self.models_list.append(self.gain_unit)
        self.models_list.append(self.qual_map)
        self.models_list.append(self.rvs)
        
        
        self.scaler_precision = self.gain_vector_precision + self.beta_displacement_precision       # TODO: remove it
        params_list = self._params_precisions.get_params_name_list() + ['scaler_precision']
        d = dict()
        for p in params_list:
            d[p] = getattr(self, p)
        self.set_params(**d)
        
    @staticmethod
    def check_module_in_list(name: str, excl_list: List[str] = None, incl_list: List[str] = None) -> bool:
        active_m = True
        if excl_list is not None and name in excl_list:
            active_m = False
        if incl_list is not None and name not in incl_list:
            active_m = False        
        return active_m
        
    def iter_over_enabled_tools(self, excl_list: List[str] = None, incl_list: List[str] = None, only_quant_int: bool = False):
        for m in self.models_list:
            if self.check_module_in_list(m.name, excl_list, incl_list) and m.is_enabled():
                if not only_quant_int or isinstance(m, QuantizationInterface):
                    yield m
                
    def iter_rev_over_enabled_tools(self, excl_list: List[str] = None, incl_list: List[str] = None, only_quant_int: bool = False):
        for m in reversed(self.models_list):
            if self.check_module_in_list(m.name, excl_list, incl_list) and m.is_enabled():
                if not only_quant_int or isinstance(m, QuantizationInterface):
                    yield m
                    
    def set_params(self, **kwargs) -> None:
        for m in self.models_list:
            if isinstance(m, QuantizationInterface):
                m.set_params(**kwargs)  
              
    def analyze(self, decisions: Decisions = None, excl_list: List[str] = None, incl_list: List[str] = None) -> Decisions:
        ans = Decisions()
        for m in self.iter_over_enabled_tools(excl_list, incl_list):
            ans[m.name] = m.analyze(decisions)
        return ans
            
    def quantize_resi(self, x: torch.Tensor, decisions: Decisions = None, excl_list: List[str] = None, incl_list: List[str] = None) -> torch.Tensor:
        if decisions is None:
            decisions = Decisions()
        dev = x.device
        for m in self.iter_over_enabled_tools(excl_list, incl_list):
            x = m.quantize_resi(x, decisions.get(m.name, None)).to(dev)
        return x
            
    def dequantize_resi(self, x: torch.Tensor, decisions: Decisions = None, excl_list: List[str] = None, incl_list: List[str] = None) -> torch.Tensor:
        if decisions is None:
            decisions = Decisions()     
        dev = x.device
        for m in self.iter_rev_over_enabled_tools(excl_list, incl_list):
            x = m.dequantize_resi(x, decisions.get(m.name, None)).to(dev)
        return x
    
    def __check_log(self, value: torch.Tensor) -> None:
        assert ( (value >= -2**13).all() and (value <= 2**13-1).all())
    
    def quantize_scale(self, x: torch.Tensor, decisions: Decisions = None, excl_list: List[str] = None, incl_list: List[str] = None) -> torch.Tensor:
        if decisions is None:
            decisions = Decisions()
        addition = torch.zeros_like(x)
        dev = x.device
        for m in self.iter_over_enabled_tools(excl_list, incl_list):
            addition = m.quantize_scale(addition, decisions.get(m.name, None)).to(dev)
        self.__check_log(addition)
        return x + addition

    def dequantize_scale(self, x: torch.Tensor, decisions: Decisions = None, excl_list: List[str] = None, incl_list: List[str] = None) -> torch.Tensor:
        if decisions is None:
            decisions = Decisions()        
        addition = torch.zeros_like(x)
        dev = x.device
        for m in self.iter_rev_over_enabled_tools(excl_list, incl_list):
            addition = m.dequantize_scale(addition, decisions.get(m.name, None)).to(dev)
        self.__check_log(addition)
        return x + addition
        
    def encode(self, ec: ECModule, decision: Decisions, *args, **kwargs) -> None:
        for m in self.iter_over_enabled_tools(only_quant_int=True):
            m.encode(ec, decision.get(m.name), *args, **kwargs)
            
    def decode(self, ec: ECModule, *args, **kwargs) -> Decisions:
        ans = Decisions()
        for m in self.iter_over_enabled_tools(only_quant_int=True):
            ans[m.name] = m.decode(ec, *args, **kwargs)
        return ans
