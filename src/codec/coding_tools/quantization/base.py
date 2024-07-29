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
from src.codec.coding_tools.interfaces import CoderEngine
from src.codec.common import Decisions, Log2LinConvertion

class QuantizationInterface(CoderEngine):
   
    @property
    def beta_displacement(self) -> float:
        return self.owner.get_owner_param('beta_displacement')
       
    @property
    def beta_displacement_log(self) -> int:
        return self.owner.get_owner_param('beta_displacement_log')
    
    @property
    def scaled_sigma_precision(self) -> int:
        return self.owner.scaled_sigma_precision

    # Virtual functions
    def _beta_displacement_log_updated(self, value: int):
        """ It calls when a new beta_displacement_log is assigned 
        """
        pass

    def set_params(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def convert_difflog2lin(self, tensor_log_domain: torch.Tensor) -> torch.Tensor:
        return Log2LinConvertion.difflog2lin(tensor_log_domain, self.sigma_precision, self.log_k)

    
    def analyze(self, decisions: Decisions) -> Decisions:
        """Analyzing input data to make a decision about module's parameters
        """
        return Decisions()
    
    def forward(self, x: torch.Tensor, decisions: Decisions = None) -> torch.Tensor:
        x = self.quantize_resi(x, decisions)
        return self.dequantize_resi(x, decisions)
    
    def quantize_resi(self, x: torch.Tensor, decisions: Decisions = None) -> torch.Tensor:
        """Quantization in linear domain

        Args:
            x (torch.Tensor): input data

        Returns:
            torch.Tensor: quantized data
        """
        raise NotImplementedError
    
    def quantize_scale(self, x: torch.Tensor, decisions: Decisions = None) -> torch.Tensor:
        """Quantization in log domain

        Args:
            x (torch.Tensor): input data

        Returns:
            torch.Tensor: quantized data
        """        
        raise NotImplementedError

    def dequantize_resi(self, x: torch.Tensor, decisions: Decisions = None) -> torch.Tensor:
        """Dequantization in linear domain

        Args:
            x (torch.Tensor): input data

        Returns:
            torch.Tensor: dequantized data
        """
        raise NotImplementedError
    
    def dequantize_scale(self, x: torch.Tensor, decisions: Decisions = None) -> torch.Tensor:
        """Dequantization in log domain

        Args:
            x (torch.Tensor): input data

        Returns:
            torch.Tensor: dequantized data
        """        
        raise NotImplementedError

    def encode(self, ec, decision: Decisions, *args, **kwargs) -> None:
        """
        Encode decisions to bitstream
        """
        pass

    # The first stage on decoder-side: decoding latent information from bitstream
    def decode(self, ec, *args, **kwargs) -> Decisions:
        """
        Decode decision from bitstream
        """
        return Decisions()