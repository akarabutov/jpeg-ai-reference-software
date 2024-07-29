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
import torch.nn as nn
from src.codec.components.base_layers import conv3x3, conv1x1
from .fusion_pred_net import FusionPredNet

def chs2group(chs):
    assert chs % 32 == 0
    return max(1, chs // 32)

class MCM_phase_base(nn.Module):
    def __init__(self, chs: int, double_input_channels=True):
        super(MCM_phase_base, self).__init__()
        self.chs = chs
        self.fusion_pred_net = FusionPredNet(chs, double_input_channels)
        self.conv = None
        
    def forward(self, pred_explicit: torch.Tensor, prev_reco: torch.Tensor = None) -> torch.Tensor:
        """Process of all input tensors on the current phase.

        Args:
            pred_explicit (torch.Tensor): information got from Hyper decoder. It has 2C channels.
            prev_reco (torch.Tensor): previously reconstructed residuals. Might be None if it is the first stage.

        Raises:
            NotImplementedError: It is pure virtual function

        Returns:
            torch.Tensor: Processed 
        """
        prev_reco = self.conv(prev_reco)
        fp_input = torch.cat( (prev_reco, pred_explicit), dim=1 )
        return self.fusion_pred_net(fp_input)
        
class MCM_phase0(MCM_phase_base):
    """Implementation phases 1 on figure E.5
    """
    def __init__(self, chs: int):
        super(MCM_phase0, self).__init__(chs, False)
    
    def forward(self, pred_explicit: torch.Tensor, prev_reco: torch.Tensor = None) -> torch.Tensor:
        return self.fusion_pred_net(pred_explicit)


class MCM_phase1(MCM_phase_base):
    """Implementation phases 2 on figure E.5
    """
    def __init__(self, chs: int):
        super(MCM_phase1, self).__init__(chs)
        self.conv = nn.Sequential(
                conv1x1(chs, chs),
                conv3x3(chs, chs, stride=1, groups=chs2group(chs))
            )
    
class MCM_phase2(MCM_phase_base):
    """Implementation phases 3 on figure E.5
    """
    def __init__(self, chs: int):
        super(MCM_phase2, self).__init__(chs)
        self.conv = nn.Sequential(
                conv1x1(2 * chs, chs),
                conv3x3(chs, chs, stride=1, groups=chs2group(chs))
            )
    
class MCM_phase3(MCM_phase_base):
    """Implementation phases 4 on figure E.5
    """
    def __init__(self, chs: int):
        super(MCM_phase3, self).__init__(chs)
        self.conv = nn.Sequential(
                conv1x1(3 * chs, chs),
                conv3x3(chs, chs, stride=1, groups=chs2group(chs))
            )
    
