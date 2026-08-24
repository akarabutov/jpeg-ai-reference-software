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
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None and -1 not in opt['gpu_ids'] else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warmup
        lr_groups_l: list for lr_groups. each for a optimizer"""
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler"""
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        for scheduler in self.schedulers:
            scheduler.step()
        # set up warm-up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return [param_group['lr'] for param_group in self.optimizers[0].param_groups]

    def get_network_description(self, network):
        """Get the string and total parameters of the network"""
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        return str(network), sum(map(lambda x: x.numel(), network.parameters()))

    def save_network(self, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)

    def save_training_state(self, epoch, iter_step):
        """Save training state during training, which will be used for resuming"""
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        """Resume the optimizers and schedulers for training"""
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)
