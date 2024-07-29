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

import os
import torch
import numpy as np
from typing import Tuple

from src.codec.entropy_coding import ECModule
from src.codec.common import Decisions
from ..base import QuantizationInterface
from .params import GainParams

class GainUnit(QuantizationInterface):
    tiny_val = 1e-9

    def __init__(self, chs=192, log_k=(np.log(54.82)-np.log(0.11))/31, *args, **kwargs):
        super(GainUnit, self).__init__(signal_enabled_flag=False, has_enabled_flag=True, enabled=1, *args, **kwargs)
       
        self.chs = chs
        self.log_k = log_k
        self.qp_num = None

        self._params_gain_unit = GainParams()
               
    def _params_loaded(self) -> None:
        qp_num = len(self.beta_list)
        if self.qp_num is None:
            self.c = torch.nn.Parameter(torch.ones(
                self.chs, qp_num))  # vectored parameter denotes multiple channels.
            self.qp_num = qp_num
            self.set_params()
        else:
            assert self.qp_num == qp_num
    
    def get_gain_vector_log(self):
        is_c_equal_1 = ((self.c - 1.0).abs() < 1E-5).all()
        if is_c_equal_1:
            vec_idx = 0
            self.logger.warn(r'Beta coefficients are not initialized')
        else:
            vec_idx, max_idx = self._get_min_and_max_beta()
            assert vec_idx == max_idx
        #base_model_beta = self.get_owner_param('base_model_beta')
        #vec_idx = self.beta_list.index(str(base_model_beta))-1
        gain_vector_log = (self.c[:, vec_idx] * (2 ** self.sigma_precision)).round().int()
        gain_vector_log = gain_vector_log.clamp_min(-(2 ** (self.gain_vector_log_bitdepth - 1)))
        return gain_vector_log
                          
    def _beta_displacement_log_updated(self, value: int):
        gain_vector_log = self.get_gain_vector_log()

        assert ( -(2 ** (self.gain_vector_log_bitdepth - 1)) <= (gain_vector_log) ).all()
        assert ( (gain_vector_log) <= (2 ** ( self.gain_vector_log_bitdepth - 1)) - 1 ).all()
        
        self.scaler_vec_log = (gain_vector_log + value).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        m = (2 ** self.scaler_precision)
        t_dev = self.scaler_vec_log.device
        self.scaler_vec = (self.convert_difflog2lin(self.scaler_vec_log.cpu()) * m).round() / m      
        self.scaler_vec = self.scaler_vec.to(t_dev)

    def export_models(self, output_dir: str, opset_version: int):
        gain_vector_log = self.get_gain_vector_log()
        v = gain_vector_log.cpu().numpy()
        os.makedirs(output_dir, exist_ok=True)
        np.savetxt(os.path.join(output_dir, "gain_unit_mlog.csv"), v, fmt='%d', delimiter=',')


    def quantize_resi(self, x: torch.Tensor, decisions: Decisions = None) -> torch.Tensor:   
        y = x * self.scaler_vec 
        return y
    

    def quantize_scale(self, x: torch.Tensor, decisions: Decisions = None) -> torch.Tensor:
        y = x + self.scaler_vec_log
        return y


    def dequantize_resi(self, x: torch.Tensor, decisions: Decisions = None) -> torch.Tensor:
        y = x  / (self.scaler_vec + self.tiny_val)
        return y


    def _get_min_and_max_beta(self) -> Tuple[int, int]:
        """generate max and min beta value, geberate the order of chosen gain vector

        Returns:
            min_n (int): minimum order of gain vector.
            max_n (int): maximum order of gain vector.
        """
        min_n, max_n = 0, len(self.beta_list) - 1

        mean_arr = torch.mean(self.c, -2)

        if (mean_arr == 0).all():
            return 0, 0

        for i in range(len(self.beta_list) - 1):
            if mean_arr[i] == 0 and mean_arr[i + 1] != 0:
                min_n = i + 1
            if mean_arr[i] != 0 and mean_arr[i + 1] == 0:
                max_n = i

        return min_n, max_n
    
    def encode(self, ec: ECModule, decision: Decisions, *args, **kwargs) -> None:
        pass
    
    def decode(self, ec: ECModule, *args, **kwargs) -> Decisions:
        return Decisions()