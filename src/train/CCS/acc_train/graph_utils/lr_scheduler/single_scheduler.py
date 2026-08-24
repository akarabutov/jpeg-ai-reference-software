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
import torch.distributed as dist
from torch.optim.lr_scheduler import MultiStepLR

from .scheduler_with_warmup import warmup_consineAnnealingLR, warmup_stepLR

__all__ = ['set_single_lr_scheduler']


def set_single_lr_scheduler(optimizer, base_batch_size, train_loader, args):
    real_batch = train_loader.batch_size * dist.get_world_size()
    real_lr = 1.0 * real_batch / base_batch_size * args.lr
    for _, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = real_lr
    if args.lr_type == 'warmup_step.step':
        scheduler = warmup_stepLR(optimizer,
                                  base_batch=base_batch_size,
                                  gamma=0.5,
                                  data_loader=train_loader,
                                  args=args)
    elif args.lr_type == 'warmup_anneal.step':
        scheduler = warmup_consineAnnealingLR(optimizer,
                                              base_batch=base_batch_size,
                                              data_loader=train_loader,
                                              args=args)
    elif args.lr_type == 'step.epoch':
        lr_steps = [int(x) for x in args.lr_steps.split(',')]
        scheduler = MultiStepLR(optimizer, milestones=lr_steps, gamma=0.5)
    elif args.lr_type == 'reduce_on_plateau.epoch':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=args.patience, factor=args.factor,
            verbose=True)  # dynamic lr scheduler
    else:
        raise ValueError('args.lr_type value error')
    return scheduler
