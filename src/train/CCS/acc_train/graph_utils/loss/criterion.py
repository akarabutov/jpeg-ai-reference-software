# The copyright in this software is being made available under the BSD
# License, included below. This software may be subject to other third party
# and contributor rights, including patent rights, and no such rights are
# granted under this license.
#
# Copyright (c) 2010-2026, ITU/ISO/IEC
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
# DISCLAIMER
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
import torch.distributed as dist
import torch.nn as nn

from src.train.CCS.metrics.ms_ssim import MSSSIM

from .regulation import Regulation

__all__ = ['set_criterion']


def _default_device():
    if torch.cuda.is_available():
        return torch.device('cuda', dist.get_rank() if dist.is_available() and dist.is_initialized() else 0)
    return torch.device('cpu')


def set_criterion(reg_loss_weight=5.0e-9, device=None):
    """Create the loss functions used by the training.

    Args:
        reg_loss_weight (float): weight of the L1 regulation loss.
        device (torch.device, optional): device of the losses. Defaults to the GPU of this process
            (or CPU when CUDA is not available).
    """
    device = _default_device() if device is None else device
    criterion = dict()
    criterion['mse'] = nn.MSELoss().to(device)
    criterion['msssim'] = MSSSIM().to(device)
    criterion['reg'] = Regulation(factor=reg_loss_weight, reg_type='L1')
    criterion['kd'] = nn.MSELoss().to(device)
    return criterion
