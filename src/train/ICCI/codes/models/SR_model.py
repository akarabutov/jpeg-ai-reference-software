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

import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.loss import CharbonnierLoss, MS_SSIM_Loss

from .base_model import BaseModel

logger = logging.getLogger('base')


class SRModel(BaseModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            self.loss_type = loss_type
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            elif loss_type == 'ms':
                self.cri_pix = MS_SSIM_Loss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        if isinstance(data['LQ'], list):
            data['LQ'] = torch.stack(data['LQ'])
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_GT:
            if isinstance(data['GT'], list):
                data['GT'] = torch.stack(data['GT'])
            self.real_H = data['GT'].to(self.device)  # GT

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)
        if len(self.l_pix_w) == 1:
            l_pix = self.l_pix_w[0] * self.cri_pix(self.fake_H, self.real_H)
        elif self.opt['datasets']['train']['YUV'] == 420:
            yuv_size = list(self.real_H.size())
            H = yuv_size[2]
            W = int(yuv_size[3]/1.5)
            uvH = int(0.5*H)
            uvW = int(0.5*W)
            l_pix = self.l_pix_w[0] * self.cri_pix(self.fake_H[:,:,0:H,0:W], self.real_H[:,:,0:H,0:W]) \
                       + self.l_pix_w[1] * self.cri_pix(self.fake_H[:,:,0:uvH,W:W+uvW], self.real_H[:,:,0:uvH,W:W+uvW]) \
                       + self.l_pix_w[2] * self.cri_pix(self.fake_H[:,:,uvH:H,W:W+uvW], self.real_H[:,:,uvH:H,W:W+uvW])
        elif self.opt['datasets']['train']['YUV'] == 444:
            # print("-------")
            # print(self.fake_H[:,0:1,:,:].shape)
            if self.loss_type == 'l2':
                l_pix = self.l_pix_w[0] * self.cri_pix(self.fake_H[:,0:1,:,:], self.real_H[:,0:1,:,:]) \
                        + self.l_pix_w[1] * self.cri_pix(self.fake_H[:,1:2,:,:], self.real_H[:,1:2,:,:]) \
                        + self.l_pix_w[2] * self.cri_pix(self.fake_H[:,2:3,:,:], self.real_H[:,2:3,:,:])
            elif self.loss_type == 'ms':
                l_pix = self.cri_pix(self.fake_H[:,0:1,:,:], self.real_H[:,0:1,:,:])
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(self.loss_type))
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test16(self):
        self.netG.eval()
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.half()
        else:
            self.netG.half()
        self.var_L = self.var_L.half()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()
    
    def test_with_padding(self, image_format = 420, padding_size = 16):
        self.netG.eval()
        with torch.no_grad():
            v2np = self.var_L.data.cpu().numpy()
            if image_format == 420:
                H = v2np.shape[2]
                W = int(v2np.shape[3]//1.5)
                Y = np.pad(v2np[:,:,0:H,0:W], pad_width=((0,0),(0,0),(padding_size*2,padding_size*2),(padding_size*2,padding_size*2)), mode='symmetric')
                U = np.pad(v2np[:,:,0:H//2,W:W+W//2], pad_width=((0,0),(0,0),(padding_size,padding_size),(padding_size,padding_size)), mode='symmetric')
                V = np.pad(v2np[:,:,H//2:H,W:W+W//2], pad_width=((0,0),(0,0),(padding_size,padding_size),(padding_size,padding_size)), mode='symmetric')
                tfnp = np.concatenate((Y, np.concatenate((U, V), axis=2)), axis=3)
                ret = torch.Tensor(tfnp).to(self.device)
            else:
                tfnp = np.pad(v2np, pad_width=((0,0),(0,0),(padding_size,padding_size),(padding_size,padding_size)), mode='symmetric')
                ret = torch.Tensor(tfnp).to(self.device)
            output_pad = self.netG(ret).data.cpu().numpy()
            if image_format == 420:
                H = output_pad.shape[2]
                W = int(output_pad.shape[3]//1.5)
                Y = output_pad[:,:,0:H,0:W]
                U = output_pad[:,:,0:H//2,W:W+W//2]
                V = output_pad[:,:,H//2:H,W:W+W//2]
                Y_crop = Y[:,:,2*padding_size:-2*padding_size,2*padding_size:-2*padding_size]
                U_crop = U[:,:,padding_size:-1*padding_size,padding_size:-1*padding_size]
                V_crop = V[:,:,padding_size:-1*padding_size,padding_size:-1*padding_size]
                output_crop = np.concatenate((Y_crop, np.concatenate((U_crop, V_crop), axis=2)), axis=3)
                output_crop = torch.Tensor(output_crop).to(self.device)
            else:    
                output_crop = output_pad[:,:,padding_size:-1*padding_size,padding_size:-1*padding_size]
                output_crop = torch.Tensor(output_crop).to(self.device)
            self.fake_H = output_crop
        self.netG.train()

    def test_x8(self, image_format = 420):
        # from https://github.com/thstkdgus35/EDSR-PyTorch
        self.netG.eval()

        def _transform(v, op, image_format):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if image_format == 420:
                H = v2np.shape[2]
                W = int(v2np.shape[3]//1.5)
                Y = v2np[:,:,0:H,0:W]
                U = v2np[:,:,0:H//2,W:W+W//2]
                V = v2np[:,:,H//2:H,W:W+W//2]
                if op == 'v':
                    Y = Y[:, :, :, ::-1].copy()
                    U = U[:, :, :, ::-1].copy()
                    V = V[:, :, :, ::-1].copy()
                elif op == 'h':
                    Y = Y[:, :, ::-1, :].copy()
                    U = U[:, :, ::-1, :].copy()
                    V = V[:, :, ::-1, :].copy()
                elif op == 't':
                    Y = Y.transpose((0, 1, 3, 2)).copy()
                    U = U.transpose((0, 1, 3, 2)).copy()
                    V = V.transpose((0, 1, 3, 2)).copy()
                tfnp = np.concatenate((Y, np.concatenate((U, V), axis=2)), axis=3) 
            else:
                if op == 'v':
                    tfnp = v2np[:, :, :, ::-1].copy()
                elif op == 'h':
                    tfnp = v2np[:, :, ::-1, :].copy()
                elif op == 't':
                    tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
                # if self.precision == 'half': ret = ret.half()
            return ret

        lr_list = [self.var_L]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf, image_format) for t in lr_list])
        with torch.no_grad():
            sr_list = [self.netG(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't', image_format)
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h', image_format)
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v', image_format)

        output_cat = torch.cat(sr_list, dim=0)
        self.fake_H = output_cat.mean(dim=0, keepdim=True)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
