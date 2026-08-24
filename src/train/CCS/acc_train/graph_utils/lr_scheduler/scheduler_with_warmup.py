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

import math

import torch.distributed as dist
from torch.optim.lr_scheduler import _LRScheduler

__all__ = ['warmup_stepLR', 'warmup_consineAnnealingLR']


class WarmupLR(_LRScheduler):
    def __init__(self,
                 scheduler,
                 pretrain_lr=1e-4,
                 init_lr=1e-3,
                 num_warmup=1,
                 warmup_strategy='linear'):
        if warmup_strategy not in ['linear', 'cos', 'constant']:
            raise ValueError(
                "Expect warmup_strategy to be one of ['linear', 'cos', 'constant'] but got {}".
                format(warmup_strategy))
        self._scheduler = scheduler
        self._init_lr = init_lr
        self._pretrain_lr = pretrain_lr
        self._num_warmup = num_warmup
        self._step_count = 0
        # Define the strategy to warm up learning rate
        self._warmup_strategy = warmup_strategy
        if warmup_strategy == 'cos':
            self._warmup_func = self._warmup_cos
        elif warmup_strategy == 'linear':
            self._warmup_func = self._warmup_linear
        else:
            self._warmup_func = self._warmup_const
        # save initial learning rate of each param group
        # only useful when each param groups having different learning rate
        self._format_param()

    def __getattr__(self, name):
        return getattr(self._scheduler, name)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        wrapper_state_dict = {
            key: value
            for key, value in self.__dict__.items() if (key != 'optimizer' and key != '_scheduler')
        }
        wrapped_state_dict = {
            key: value
            for key, value in self._scheduler.__dict__.items() if key != 'optimizer'
        }
        return {'wrapped': wrapped_state_dict, 'wrapper': wrapper_state_dict}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict['wrapper'])
        self._scheduler.__dict__.update(state_dict['wrapped'])

    def _format_param(self):
        # learning rate of each param group will increase
        # from the min_lr to initial_lr
        for group in self._scheduler.optimizer.param_groups:
            group['warmup_max_lr'] = group['lr']
            group['warmup_initial_lr'] = min(self._init_lr, group['lr'])

    def _warmup_cos(self, start, end, pct):
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def _warmup_const(self, start, end, pct):
        return start if pct < 0.9999 else end

    def _warmup_linear(self, start, end, pct):
        return (end - start) * pct + start

    def get_lr(self):
        lrs = []
        step_num = self._step_count
        # warm up learning rate
        if step_num <= self._num_warmup:
            for group in self._scheduler.optimizer.param_groups:
                computed_lr = self._warmup_func(group['warmup_initial_lr'], group['warmup_max_lr'],
                                                step_num / self._num_warmup)
                lrs.append(computed_lr)
        else:
            lrs = self._scheduler.get_lr()
        return lrs

    def step(self, is_pretrain=False):
        if is_pretrain:
            for param_group in self._scheduler.optimizer.param_groups:
                param_group['lr'] = self._pretrain_lr
            return
        if self._step_count <= self._num_warmup:
            values = self.get_lr()
            for param_group, lr in zip(self._scheduler.optimizer.param_groups, values):
                param_group['lr'] = lr
        else:
            self._scheduler.step()
        self._step_count += 1


def warmup_consineAnnealingLR(optimizer, base_batch, data_loader, args):
    from torch.optim.lr_scheduler import CosineAnnealingLR
    real_batch = data_loader.batch_size * dist.get_world_size()
    # real_lr = 1.0 * real_batch / base_batch * args.lr
    warmup_steps = int(args.base_warmup_epoch * len(data_loader) * real_batch / base_batch)
    anneal_steps = args.epochs * len(data_loader) - warmup_steps
    scheduler = WarmupLR(CosineAnnealingLR(optimizer, anneal_steps, eta_min=args.anneal_final_lr),
                         pretrain_lr=args.lr,
                         init_lr=args.lr / 10.0,
                         num_warmup=warmup_steps,
                         warmup_strategy='linear')
    return scheduler


def warmup_stepLR(optimizer, base_batch, gamma, data_loader, args):
    from torch.optim.lr_scheduler import MultiStepLR
    real_batch = data_loader.batch_size * dist.get_world_size()
    # real_lr = 1.0 * real_batch / base_batch * args.lr
    warmup_steps = int(args.base_warmup_epoch * len(data_loader) * real_batch / base_batch)
    lr_steps = [int(x) for x in args.lr_steps.split(',')]
    milestones = [item * len(data_loader) - warmup_steps for item in lr_steps]
    scheduler = WarmupLR(MultiStepLR(optimizer, milestones=milestones, gamma=0.5),
                         pretrain_lr=args.lr,
                         init_lr=args.lr / 10.0,
                         num_warmup=warmup_steps,
                         warmup_strategy='linear')
    return scheduler


if __name__ == '__main__':
    import torch
    from torch.optim.lr_scheduler import MultiStepLR
    print('test WarmupLR')
    p1 = torch.nn.Parameter(torch.arange(10, dtype=torch.float32))
    p2 = torch.nn.Parameter(torch.arange(10, dtype=torch.float32))
    init_lr = 0.00001
    target_lr = 0.1
    # params = [{'params': [p1]}, {'params': [p2], 'lr': 0.001}]
    params = [{'params': [p1]}, {'params': [p2]}]
    optimizer = torch.optim.Adam(params, lr=target_lr, weight_decay=0.0)
    scheduler = MultiStepLR(optimizer, milestones=[3, 5], gamma=0.5)
    # scheduler.step()
    # optimizer.step()
    len_train_loader = 200
    warmup_steps = len_train_loader * 5
    milestones = [item * len_train_loader - warmup_steps for item in [6, 8]]
    print(milestones)
    scheduler = WarmupLR(MultiStepLR(optimizer, milestones=milestones, gamma=0.5),
                         init_lr=init_lr,
                         num_warmup=warmup_steps,
                         warmup_strategy='linear')
    total_step = 0
    for epoch in range(11):
        for step in range(len_train_loader):
            optimizer.step()
            scheduler.step()
            for id_group, param_group in enumerate(optimizer.param_groups):
                lr = param_group['lr']
                print('total_step', total_step, 'id_group', id_group, 'epoch', epoch, 'step', step,
                      'lr', lr)
            total_step += 1
