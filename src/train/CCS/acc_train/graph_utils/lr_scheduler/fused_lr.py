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

import copy

from ..optimizer.fused_opt import FusedOpt
from .set_scheduler import set_single_lr_scheduler

__all__ = ['set_fused_lr_scheduler']


class FusedLr():
    def __init__(self, fused_opt, base_batch_size, train_loader, args):
        lamb_args = copy.deepcopy(args)
        self.lamb_lr = set_single_lr_scheduler(fused_opt.lamb_opt, base_batch_size, train_loader,
                                               lamb_args)

        adam_args = copy.deepcopy(args)
        adam_args.lr = adam_args.lr / 10.0
        adam_args.anneal_final_lr = adam_args.anneal_final_lr / 10.0
        adam_args.warmup_init_lr = adam_args.lr / 10.0
        adam_args.lr_type = adam_args.adam_lr_type
        self.adam_lr = set_single_lr_scheduler(fused_opt.adam_opt, base_batch_size, train_loader,
                                               adam_args)

    def step(self):
        self.adam_lr.step()
        self.lamb_lr.step()


def set_fused_lr_scheduler(optimizer, base_batch_size, train_loader, args):
    assert isinstance(optimizer, FusedOpt)
    return FusedLr(optimizer, base_batch_size, train_loader, args)
