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
import random
import math

import einops
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import collections
from functools import partial
from torch.nn.parallel import DistributedDataParallel as DDP

from src.train.CCS.utils import save_checkpoint, weights_init

try:
    from torch.cuda.amp import autocast
except ImportError:
    raise ImportError('PyTorch 1.8.x is needed')
from src.codec.common.colorspace import ColorSpace
from src.codec.components.contexts.context import Context
from src.codec.components.contexts.utils import Upsample_proc
from src.codec.components.entropy_coding.entropy_estimation import cal_y_likelihoods_decoupled
from src.codec.components.entropy_coding.prob_models.factorized import \
    FactorizedProbModel
from src.codec.components.entropy_coding.prob_models.gm import GMProbModel
from src.codec.components.entropy_coding.prob_models.laplacian import LaplacianProbModel
from src.codec.components.vr_quantizers.vrq_vec import VrqVec
from src.codec.components import (
        EncoderFactory,
        DecoderFactory,
        HyperEncoderFactory,
        HyperDecoderFactory,
        HyperScaleDecoderFactory
    )
from src.codec.coding_tools.tiling import TileManager
from src.codec.common import tiling
from typing import Tuple, List, Union

__all__ = ['Net']


def get_beta_list(args):
    beta_list = [float(x) for x in args.beta_list.split(',')]
    if dist.get_rank() == 0:
        print('beta list for current train =', beta_list)
    model_beta_list = [
        0.0005, 0.001, 0.002, 0.004, 0.007, 0.01, 0.012, 0.015, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5, 0.75, 1.0, 2.0, 3.0
    ]
    return beta_list, model_beta_list


def getStat(feature, stat, first_pass=True):
    n,c,h,w = feature.shape
    if first_pass:
        if stat['MaxList'] == []:
            for i in range(c):
                stat['MaxList'].append(torch.max(feature[:,i,:,:]).cpu().item())
                stat['MinList'].append(torch.min(feature[:,i,:,:]).cpu().item())
        else:
            for i in range(c):
                stat['MaxList'][i] = max(torch.max(feature[:,i,:,:]).cpu().item(), stat['MaxList'][i])
                stat['MinList'][i] = min(torch.min(feature[:,i,:,:]).cpu().item(), stat['MinList'][i])
    else:
        bin_num = 1000
        if stat['StaList'] == []:
            stat['StaList'] = np.zeros((c, bin_num)).tolist()
        for i in range(c):
            fea = feature[:,i,:,:].contiguous().view(-1)
            cnts = torch.histc(fea, bins=bin_num, min=stat['MinList'][i], max=stat['MaxList'][i])
            stat['StaList'][i] = np.sum([stat['StaList'][i], cnts.cpu().tolist()], axis=0).tolist()

class DummyDDP(nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(DummyDDP, self).__init__()
        self.module = module
        
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class MultiCoder(nn.Module):
    def __init__(self, factory: Union[EncoderFactory, DecoderFactory], names_list: List[str], *args, **kwargs):
        super(MultiCoder, self).__init__()
        self.coders = nn.ModuleDict()
        for name in names_list:
            self.coders.add_module(name, factory.create_instance(
                name=name,
                *args, **kwargs
            ))
    
    def count(self):
        return len(self.coders)        
    
    def forward(self, img: torch.Tensor, h:int, w:int, *args, **kwargs) -> Tuple[torch.Tensor]:
        ans = list()
        is_collect = kwargs.get('is_collect', False)
        if not is_collect:
            for coder in self.coders.values():
                ans.append(coder(img, h=h, w=w, *args, **kwargs)[..., :h, :w])
        else:
            stat = kwargs.pop('stat', None)
            first_pass = kwargs.pop('first_pass', True)
            for name, coder in self.coders.items():
                x1, x2, x3, x4, x = coder(img, h=h, w=w, *args, **kwargs)
                ans.append(x[..., :h, :w])
                if 'bop' in name:
                    getStat(x1, stat['E1B'], first_pass)
                    getStat(x2, stat['E2B'], first_pass)
                    getStat(x3, stat['E3B'], first_pass)
                    getStat(x4, stat['E4B'], first_pass)
                    getStat(x, stat['E5B'], first_pass)
                elif 'hop' in name:
                    getStat(x1, stat['E1H'], first_pass)
                    getStat(x2, stat['E2H'], first_pass)
                    getStat(x3, stat['E3H'], first_pass)
                    getStat(x4, stat['E4H'], first_pass)
                    getStat(x, stat['E5H'], first_pass)
            return torch.cat(ans, dim=0), stat
        return torch.cat(ans, dim=0)

class Net(nn.Module):
    """AI Codec Net"""
    def __init__(self, args):
        super(Net, self).__init__()
        N, N_UV, N_Y, _ = args.N, args.N_UV, args.N_Y, 1
        self.args = args
        self.mem_format = 'channel_first'
        self.device = torch.device(args.local_rank if torch.cuda.is_available() else 'cpu')
        self.beta_list, self.model_beta_list = get_beta_list(args)
        self.vae_encoder_type_list = args.vae_encoder_type_list
        self.vae_decoder_type_list = args.vae_decoder_type_list

        self.sigma_min = args.sigma_quant_min
        self.sigma_max = args.sigma_quant_max
        self.sigma_level = args.sigma_quant_level
        self.log_k = (np.log(self.sigma_max) - np.log(self.sigma_min)) / (self.sigma_level - 1)
        self.log_b = np.log(self.sigma_min)

        self.encoder_Y = MultiCoder(EncoderFactory(), 
                                    chs_ls=N,
                                    names_list=[f"{x}_prim" for x in args.vae_encoder_type_list]
                                    )
        
        self.encoder_UV = MultiCoder(EncoderFactory(), 
                                    chs_ls=N_UV,
                                    names_list=[f"{x}_sec" for x in args.vae_encoder_type_list]
                                    )
        
        self.vr_vec_Y = VrqVec(chs=N + N_Y, qp_num=len(self.model_beta_list), log_k=self.log_k)
        self.vr_vec_UV = VrqVec(chs=N_UV, qp_num=len(self.model_beta_list), log_k=self.log_k)
        self.decoder_Y = MultiCoder(DecoderFactory(),
                                    chs_ls=N,
                                    names_list=[f"{x}_prim" for x in args.vae_decoder_type_list]
            )
        self.decoder_UV = MultiCoder(DecoderFactory(),
                                     chs_ls=N_UV,
                                     names_list=[f"{x}_sec" for x in args.vae_decoder_type_list]
            )
        self.total_loss_comp = len(args.vae_decoder_type_list) * len(args.vae_encoder_type_list)
        if args.entropy == 'gaussian':
            # self.entropy = SSGMProbModel(scale_bound=args.scale_bound)
            self.entropy = GMProbModel(scale_table=None, 
                                       scale_level=args.sigma_quant_level, 
                                       scale_max=args.sigma_quant_max,
                                       scale_min=args.sigma_quant_min,
                                       bound_offset=args.sigma_bound_offset)
        else:
            self.entropy = LaplacianProbModel(scale_bound=args.scale_bound)

        self.hyper_encoder_Y = HyperEncoderFactory().create_instance(
                name=args.hyper_encoder_type,
                chs=N
            )
        self.hyper_encoder_UV = HyperEncoderFactory().create_instance(
                name=args.hyper_encoder_type,
                chs=N_UV,
                skip_depth_step=True
            )
        self.hyper_decoder_Y = HyperDecoderFactory().create_instance(
            name=args.hyper_decoder_type,
            chs=N,
            num_out=1
        )
        self.hyper_decoder_UV = HyperDecoderFactory().create_instance(
            name=args.hyper_decoder_type,
            chs=N_UV,
            num_out=1,
            skip_depth_step=True
        )
        self.hyper_scale_decoder_Y = HyperScaleDecoderFactory().create_instance(
            name=args.hyper_scale_decoder_type,
            chs=N
        )
        self.hyper_scale_decoder_UV = HyperScaleDecoderFactory().create_instance(
            name=args.hyper_scale_decoder_type,
            chs=N_UV,
            skip_depth_step=True
        )
        
        self.hyper_scale_decoder_Y. sigma_idx_max_value = self.sigma_level - 1
        self.hyper_scale_decoder_UV.sigma_idx_max_value = self.sigma_level - 1
        
        self.hyper_entropy_Y = FactorizedProbModel(channels=N, max_symbol=62)  # fixed the hyper bottleneck
        self.hyper_entropy_UV = FactorizedProbModel(channels=N_UV, max_symbol=62)
        self.context_Y = Context(chs=N, quantize_func=self.quantize)

        self.register_buffer('channel_wise_entropy_Y', torch.zeros(N, dtype=torch.float32, device=self.device))
        self.register_buffer('channel_wise_entropy_UV', torch.zeros(N_UV, dtype=torch.float32, device=self.device))

        self.model_list = [
            self.encoder_Y, self.encoder_UV, self.vr_vec_Y, self.vr_vec_UV,
            self.decoder_Y, self.decoder_UV, self.entropy, self.hyper_encoder_Y,
            self.hyper_encoder_UV, self.hyper_decoder_Y, self.hyper_decoder_UV,
            self.hyper_scale_decoder_Y, self.hyper_scale_decoder_UV, self.hyper_entropy_Y,
            self.hyper_entropy_UV, self.context_Y
        ]

        self.named_model_list = [('encoder_Y', self.encoder_Y), ('encoder_UV', self.encoder_UV),
                                 ('vr_vec_Y', self.vr_vec_Y),
                                 ('vr_vec_UV', self.vr_vec_UV),
                                 ('decoder_Y', self.decoder_Y), ('decoder_UV', self.decoder_UV),
                                 ('entropy', self.entropy),
                                 ('hyper_encoder_Y', self.hyper_encoder_Y),
                                 ('hyper_encoder_UV', self.hyper_encoder_UV),
                                 ('hyper_decoder_Y', self.hyper_decoder_Y),
                                 ('hyper_decoder_UV', self.hyper_decoder_UV),
                                 ('hyper_scale_decoder_Y', self.hyper_scale_decoder_Y),
                                 ('hyper_scale_decoder_UV', self.hyper_scale_decoder_UV),
                                 ('hyper_entropy_Y', self.hyper_entropy_Y),
                                 ('hyper_entropy_UV', self.hyper_entropy_UV),
                                 ('context_Y', self.context_Y)]

        self._init_weight()
        self._to_device()
        self.loss_weights = self._loss_weight_parse(args.loss_weights)
        enc_count = len(args.vae_encoder_type_list)
        self.loss_weights_summ = torch.zeros([enc_count])
        for enc_num in range(enc_count):
            self.loss_weights_summ[enc_num] = self.loss_weights[enc_num::enc_count].sum()
            
        self._register_load_state_dict_pre_hook(self._load_state_dict_hook)

    def zero_channel_wise_entropy(self):
        self.channel_wise_entropy_Y.fill_(0)
        self.channel_wise_entropy_UV.fill_(0)
        
    def _loss_weight_parse(self, weights_str: str) -> torch.Tensor:
        
        weights_arr = [[float(y) for y in x.split(',')] for x in weights_str.split(';')]
        
        return torch.tensor(weights_arr, dtype=torch.float, device=self.device).flatten()
        

            
    @staticmethod
    def parse_multi_params(param_str, count_items) -> List[List[int]]:
        ans = [[int(x) for x in y.split(',')] for y in param_str.split(';')] if param_str is not None else None
        if ans is not None:
            if len(ans) != count_items:
                loop_cou = count_items - len(ans)
                for _ in range(loop_cou):
                    ans.append(ans[-1])
        return ans

    def quantize(self, data:torch.Tensor,  gvae_params:dict= None, tool_params:dict = None) -> Tuple[torch.Tensor]:
        """Quantize and dequantize residual process, including residual processing by RVS and skip tools

        Args:
            data (torch.Tensor): data for quantization
            mu (torch.Tensor): Pred_explicit.
            gvae_params (dict, optional): Gvea parameters, defaults to None.
            tool_params (dict, optional): Rvs and res_skip parameters, defaults to None..

        Returns:
            torch.Tensor: Dequantized y
            torch.Tensor: Quantized residual
        """

        if gvae_params is not None:
            vrq_func = gvae_params["gvae"]
            n_rate = gvae_params["n_rate"]
            ft = gvae_params["ft"]

        mask2_list = tool_params["mask2"]
        
        data_s = data
        
        # Vrq quantize the residual
        data_q = vrq_func(data_s, n_rate, ft) if gvae_params is not None else data_s

        # skip block and mask padding boundary
        mask2_list = mask2_list.to(torch.bool)
        data_q[~mask2_list] = 0

        type_info = torch.iinfo(torch.int32)
        data_q = torch.clamp(data_q, type_info.min, type_info.max)

        resi_rq = data_q - (data_q - data_q.round()).detach()

        # Vrq dequantize the residual
        data_dq =  vrq_func.dequantize(resi_rq, n_rate, ft) if gvae_params is not None else resi_rq

        return data_dq, resi_rq

    def set_train_mode(self):
        for model in self.model_list:
            model.train()

    def set_eval_mode(self):
        for model in self.model_list:
            model.eval()

    def _init_weight(self):
        self.encoder_Y.apply(weights_init)
        self.encoder_UV.apply(weights_init)
        self.vr_vec_Y.apply(weights_init)
        self.vr_vec_UV.apply(weights_init)
        self.decoder_Y.apply(weights_init)
        self.decoder_UV.apply(weights_init)
        self.hyper_encoder_Y.apply(weights_init)
        self.hyper_encoder_UV.apply(weights_init)
        self.hyper_decoder_Y.apply(weights_init)
        self.hyper_decoder_UV.apply(weights_init)

        self.hyper_scale_decoder_Y.apply(weights_init)
        self.hyper_scale_decoder_UV.apply(weights_init)

        self.context_Y.apply(weights_init)
        #self.context_UV.apply(weights_init)

    def _to_device(self):
        for model in self.model_list:
            model.to(self.device)

    def remove_keys(self, ckpt, keys):
        for k in keys:
            if k in ckpt:
                del ckpt[k]
        return ckpt
    
    def _load_state_dict_hook(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        case_changes = dict()
        op_dict = {
            'encoder': self.vae_encoder_type_list,
            'decoder': self.vae_decoder_type_list
        }
        comps = {
            'Y': 'prim',
            'UV': 'sec'
        }
        additional_submodule="coders"
        for k_full in state_dict.keys():
            if k_full.startswith(prefix):
                rest_key = k_full[len(prefix):]
                rest_key = rest_key.replace('.cpu_', '.sop_')           # Need only to work 
                for enc_dec_type, op_list in op_dict.items():
                    if rest_key.startswith(enc_dec_type):
                        rest_key_list = rest_key.split('.')
                        comp_subtype = ""
                        for comp, new_comp in comps.items():
                            if f"_{comp}" in rest_key_list[0]:
                                comp_subtype = new_comp
                        for op in op_list:
                            if rest_key_list[1].startswith(op):
                                new_rest_key_list = [rest_key_list[0]] + [additional_submodule] + [f"{op}_{comp_subtype}"] + rest_key_list[2:]
                                case_changes[k_full] = prefix + ".".join(new_rest_key_list)
                
        for o,n in case_changes.items():
            state_dict[n] = state_dict.pop(o)

    def resume_weight(self, ckpt_path, strict: bool = True):
        if os.path.isfile(ckpt_path):
            print("=> loading checkpoint '{}'".format(ckpt_path))
            dev = None if torch.cuda.is_available() else torch.device('cpu')
            checkpoint = torch.load(ckpt_path, map_location=dev)
        else:
            print("=> no checkpoint found at '{}'".format(ckpt_path))
            return
        epoch = checkpoint.get('epoch', -1)

        checkpoint = self.remove_keys(checkpoint, ["epoch", "best_loss", "optimizer"])
        torch.distributed.barrier()

        self.load_state_dict(checkpoint, strict)
        print("=> loaded checkpoint epoch {}".format(epoch))

    def save_model_and_opt(self,
                           args,
                           epoch,
                           optimizer,
                           best_loss,
                           model_prefix='',
                           is_best=False):
        """Save the checkpoint.
        """
        if not os.path.exists(args.train_url):
            os.makedirs(args.train_url)
        checkpoint_name = model_prefix + '.pth'
        abs_checkpoint_name = os.path.join(args.train_url, checkpoint_name)
        if self.entropy._offset.numel() <= 0:
            self.entropy.update_scale_table()
        state_dict_out = collections.OrderedDict()
        state_dict_out.update({
            'epoch': epoch + 1,
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict()
        })
        s = self.state_dict()
        for k in list(s.keys()):
            if 'module.' in k:
                new_k = k.replace('module.', '')
                s[new_k] = s.pop(k)
            
        state_dict_out.update(s)
        save_checkpoint(state_dict_out, abs_checkpoint_name, is_best=is_best)

    def _replace_batchnorm(self):
        self.encoder_Y = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_Y)
        self.encoder_UV = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_UV)
        self.vr_vec_Y = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.vr_vec_Y)
        self.vr_vec_UV = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.vr_vec_UV)
        self.decoder_Y = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.decoder_Y)
        self.decoder_UV = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.decoder_UV)
        self.entropy = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.entropy)
        self.hyper_encoder_Y = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.hyper_encoder_Y)
        self.hyper_encoder_UV = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            self.hyper_encoder_UV)
        self.hyper_decoder_Y = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.hyper_decoder_Y)
        self.hyper_decoder_UV = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            self.hyper_decoder_UV)
        self.hyper_scale_decoder_Y = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            self.hyper_scale_decoder_Y)
        self.hyper_scale_decoder_UV = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            self.hyper_scale_decoder_UV)
        self.hyper_entropy_Y = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.hyper_entropy_Y)
        self.hyper_entropy_UV = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            self.hyper_entropy_UV)
        self.context_Y = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.context_Y)
        #self.context_UV = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.context_UV)

    def DDP(self, local_rank):
        self._replace_batchnorm()
        ddp_obj = DDP if torch.cuda.device_count() > 0 else DummyDDP
        self.encoder_Y = ddp_obj(self.encoder_Y, [local_rank], local_rank)
        self.encoder_UV = ddp_obj(self.encoder_UV, [local_rank], local_rank)
        self.vr_vec_Y = ddp_obj(self.vr_vec_Y, [local_rank], local_rank)
        self.vr_vec_UV = ddp_obj(self.vr_vec_UV, [local_rank], local_rank)
        self.decoder_Y = ddp_obj(self.decoder_Y, [local_rank], local_rank)
        self.decoder_UV = ddp_obj(self.decoder_UV, [local_rank], local_rank)
        self.hyper_encoder_Y = ddp_obj(self.hyper_encoder_Y, [local_rank], local_rank)
        self.hyper_encoder_UV = ddp_obj(self.hyper_encoder_UV, [local_rank], local_rank)
        self.hyper_decoder_Y = ddp_obj(self.hyper_decoder_Y, [local_rank], local_rank)
        self.hyper_decoder_UV = ddp_obj(self.hyper_decoder_UV, [local_rank], local_rank)
        self.hyper_scale_decoder_Y = ddp_obj(self.hyper_scale_decoder_Y, [local_rank], local_rank)
        self.hyper_scale_decoder_UV = ddp_obj(self.hyper_scale_decoder_UV, [local_rank], local_rank)
        self.hyper_entropy_Y = ddp_obj(self.hyper_entropy_Y, [local_rank], local_rank)
        self.hyper_entropy_UV = ddp_obj(self.hyper_entropy_UV, [local_rank], local_rank)
        self.context_Y = ddp_obj(self.context_Y, [local_rank], local_rank)
        #self.context_UV = DDP(self.context_UV, [local_rank], local_rank)

    def parameters(self):
        analysis_models = [{
            'params': self.encoder_Y.parameters()
        }, {
            'params': self.encoder_UV.parameters()
        }]

        synthesis_models = [{
            'params': self.decoder_Y.parameters()
        }, {
            'params': self.decoder_UV.parameters()
        }]

        entropy_part_models = [{
            'params': self.hyper_encoder_Y.parameters()
        }, {
            'params': self.hyper_encoder_UV.parameters()
        }, {
            'params': self.hyper_decoder_Y.parameters()
        }, {
            'params': self.hyper_decoder_UV.parameters()
        }, {
            'params': self.hyper_scale_decoder_Y.parameters()
        }, {
            'params': self.hyper_scale_decoder_UV.parameters()
        }, {
            'params': self.hyper_entropy_Y.parameters()
        }, {
            'params': self.hyper_entropy_UV.parameters()
        }]

        entropy_part_models += [{
            'params': self.context_Y.parameters()
        }]

        gain_unit_models = [{
            'params': self.vr_vec_Y.parameters()
        }, {
            'params': self.vr_vec_UV.parameters()
        }]
        return analysis_models, synthesis_models, entropy_part_models, \
            gain_unit_models

    def convert_memory_format(self, choice='channel_first'):

        if choice == 'channel_last':
            target_format = torch.channels_last
        elif choice == 'channel_first':
            target_format = torch.contiguous_format
        else:
            raise ValueError('Unsupported memory format')

        if self.mem_format == choice:
            return

        for item in self.model_list:
            item = item.to(memory_format=target_format)
        self.mem_format = choice

    def addZNoise(self, input):
        out_hat = input.detach().round() - input.detach() + input
        intermediate = input.detach() + torch.empty_like(input).uniform_(-0.5, 0.5)
        out_for_lh_calculation = intermediate.round() - intermediate + input
        return out_hat, out_for_lh_calculation

    def addResiNoise(self, input):
        intermediate = input.detach() + torch.empty_like(input).uniform_(-0.5, 0.5)
        out = intermediate.round() - intermediate + input
        return out

    def _round_with_detach(self, input):
        input_detached = input.detach()
        detached_round = input_detached.round() - input_detached + input
        return detached_round

    def _gen_skip_cubeflag(self, res_hat, mask_map, cube_flag_thre):
        res_hat_skip = res_hat.clone()
        res_hat_skip[mask_map] = 0
        diff_yhat = (res_hat_skip - res_hat).abs()
        N,C,H,W = diff_yhat.shape
        diff_yhat = diff_yhat.reshape(1,N,C,H,W)
        cube_size = 8
        cube_chan = C
        h_pad = ((H+cube_size-1)// cube_size) * cube_size - H
        w_pad = ((W+cube_size-1)// cube_size) * cube_size - W
        diff_yhat = F.pad(diff_yhat, (0, w_pad, 0, h_pad), value=0)
        maxpool = torch.nn.MaxPool3d((cube_chan, cube_size, cube_size), (cube_chan, cube_size, cube_size), 0)
        cubeflag = (maxpool(diff_yhat) > cube_flag_thre) # skip_mask = cubeflag ! skip_masp
        cubeflag = ~cubeflag[0,:,:,:,:]
        cubeflag = einops.repeat(cubeflag, 'a b c d -> a (b repeat1) (c repeat2) (d repeat3)', repeat1=cube_chan, repeat2=cube_size, repeat3=cube_size)
        cubeflag = cubeflag[:, :C, :H, :W]
        return cubeflag

    def _skip_latent(self, res_hat, scale, likelihoods, skip_thre=0.2, cube_flag_thre=1):
        mask_map = (scale <= skip_thre)
        cubeflag = self._gen_skip_cubeflag(res_hat, mask_map, cube_flag_thre)
        mask_map = mask_map & cubeflag
        res_hat[mask_map] = 0
        likelihoods[mask_map] = 1.0
        return res_hat, likelihoods


    def _rec_from_latent_code(self, y_hat_Y, y_hat_UV, h, w, args):
        x_reco_Y = self.decoder_Y(y_hat_Y, h=h, w=w)
        y_hat_Y_ds = y_hat_Y
        y_hat_UV = torch.cat((y_hat_Y_ds, y_hat_UV), dim=1)
        x_reco_UV = self.decoder_UV(y_hat_UV, h=h, w=w)
        xY_reco = x_reco_Y
        xU_reco, xV_reco = x_reco_UV.chunk(2, dim=1)
        return xY_reco, xU_reco, xV_reco

    def _tiled_rec_from_latent_code(self, net_input, y_hat_Y, y_hat_UV, h_i, w_i, tile_manager_Y, tile_manager_UV, args):
        if tile_manager_Y.is_enabled():
            x_reco_Y = torch.empty([y_hat_Y.shape[0] * self.decoder_Y.module.count()] + list(net_input['xInput_Y'].shape[1:]), device=y_hat_Y.device)
            for tile_info in tile_manager_Y.get_iter_over_tiles(y_hat_Y, x_reco_Y):
                latent_tile_data = tile_info.get_data()
                s = tile_info.output_shape()
                image_tile_data = self.decoder_Y(latent_tile_data, h=s[0], w=s[1])
                tile_info.assign_data(image_tile_data)
        else:
            x_reco_Y = self.decoder_Y(y_hat_Y, h=h_i, w=w_i)

        y_hat_Y_ds = y_hat_Y
        y_hat_UV = torch.cat((y_hat_Y_ds, y_hat_UV), dim=1)
        if tile_manager_UV.is_enabled():
            x_reco_UV = torch.empty([y_hat_UV.shape[0] * self.decoder_UV.module.count(), 2, 2 * net_input['xInput_UV'].shape[2], 2 * net_input['xInput_UV'].shape[3]], device=y_hat_UV.device)

            for tile_info in tile_manager_UV.get_iter_over_tiles(y_hat_UV, x_reco_UV):
                latent_tile_data = tile_info.get_data()
                s = tile_info.output_shape()
                image_tile_data = self.decoder_UV(latent_tile_data, h=s[0], w=s[1])
                tile_info.assign_data(image_tile_data)
        else:
            x_reco_UV = self.decoder_UV(y_hat_UV, h=h_i, w=w_i)

        xY_reco = x_reco_Y
        xU_reco, xV_reco = x_reco_UV.chunk(2, dim=1)
        return xY_reco, xU_reco, xV_reco


    def _decode_to_loss(
            self,
            y_Y,
            y_UV,
            psi_Y,
            psi_UV,
            scale_hat_Y,
            scale_hat_UV,
            delta_Y,
            delta_UV,
            beta,
            net_input,
            criterion,
            z_likelihoods_Y,
            z_likelihoods_UV,
            args,
            amp=False,
            is_train=False
    ):

        n_rate, ft = self._nrate_n_ft(beta)
        with autocast(enabled=amp):
            mean_Y = psi_Y
            mean_UV = psi_UV
            res_Y = self.vr_vec_Y(y_Y - mean_Y, n_rate, ft)
            res_UV = self.vr_vec_UV(y_UV - mean_UV, n_rate, ft)
            if is_train:
                res_Y_for_lh_calculation = self.addResiNoise(res_Y)  # res_tilde_Y
                res_UV_for_lh_calculation = self.addResiNoise(res_UV)
            else:
                res_Y_for_lh_calculation = self._round_with_detach(res_Y)
                res_UV_for_lh_calculation = self._round_with_detach(res_UV)

        y_likelihoods_Y = cal_y_likelihoods_decoupled(
            None, None, None, self.entropy,
            scale_hat_Y.to(dtype=torch.float32),
            res=res_Y_for_lh_calculation.to(dtype=torch.float32)
        )
        y_likelihoods_UV = cal_y_likelihoods_decoupled(
            None, None, None, self.entropy,
            scale_hat_UV.to(dtype=torch.float32),
            res=res_UV_for_lh_calculation.to(dtype=torch.float32)
        )
        h, w = net_input['input_shape']['H'], net_input['input_shape']['W']
        with autocast(enabled=amp):
            # if is_train:
            res_Y = self._round_with_detach(res_Y)
            res_UV = self._round_with_detach(res_UV)

            # element wise skip in train
            if args.skip_thre > 1e-12:
                res_Y, y_likelihoods_Y = self._skip_latent(res_Y, scale_hat_Y, y_likelihoods_Y, skip_thre=args.skip_thre, cube_flag_thre=args.cube_flag_thre)
                res_UV, y_likelihoods_UV = self._skip_latent(res_UV, scale_hat_UV, y_likelihoods_UV, skip_thre=args.skip_thre, cube_flag_thre=args.cube_flag_thre)

            if args.enable_gvae:
                y_hat_Y = self.vr_vec_Y.module.dequantize(res_Y, n_rate, ft) + mean_Y
                y_hat_UV = self.vr_vec_UV.module.dequantize(res_UV, n_rate, ft) + mean_UV
            else:
                y_hat_Y = self.vr_vec_Y.module.dequantize(res_Y + mean_Y, n_rate, ft)
                y_hat_UV = self.vr_vec_UV.module.dequantize(res_UV + mean_UV, n_rate, ft)
            x_reco_Y = self.decoder_Y(y_hat_Y, h=h, w=w)
            y_hat_Y_ds = y_hat_Y
            y_hat_UV = torch.cat((y_hat_Y_ds, y_hat_UV), dim=1)
            x_reco_UV = self.decoder_UV(y_hat_UV, h=h, w=w)
            xY_reco = x_reco_Y
            xU_reco, xV_reco = x_reco_UV.chunk(2, dim=1)

        rd_loss = self._rd_loss(criterion,
                                y_likelihoods_Y,
                                y_likelihoods_UV,
                                z_likelihoods_Y,
                                z_likelihoods_UV,
                                xY_reco,
                                xU_reco,
                                xV_reco,
                                net_input,
                                beta,
                                args,
                                is_train=is_train)
        if is_train:
            reg_loss = self._regulation_loss(criterion, args, amp=amp)
            return {'rd_loss': rd_loss, 'reg_loss': reg_loss}
        else:
            return rd_loss
    
    def _index_to_scale(self, index):
        scale = torch.exp(index * self.log_k + self.log_b)
        return scale

    def get_total_loss(self, loss_dict):
        total_loss_dict = dict()
        total_weight = 0
        for key_enc, sub_loss_dict in loss_dict.items():
            assert key_enc in ['bop', 'hop']
            #if dist.get_rank() == 0:
            #    print('-->enc-key', key_enc, self.loss_weights_enc[key_enc])
            total_weight += self.loss_weights_enc[key_enc]
            for key, val in sub_loss_dict.items():
                if key in total_loss_dict:
                    total_loss_dict[key] = total_loss_dict[key] + val * self.loss_weights_enc[key_enc]
                else:
                    total_loss_dict[key] = val * self.loss_weights_enc[key_enc]
        for key in total_loss_dict.keys():
            total_loss_dict[key] = total_loss_dict[key] / total_weight
        return total_loss_dict

    def train_forward_to_loss(self, x, criterion, beta, args, amp=True):
        net_input = self._preprocess(x, args)
        n_rate, ft = self._nrate_n_ft(beta)
        xInput_Y, xInput_UV = net_input['xInput_Y'], net_input['xInput_UV']
        h, w = net_input['input_shape']['H'], net_input['input_shape']['W']
        h2, w2 = (h+1)>>1, (w+1)>>1
        with torch.set_grad_enabled('analysis' not in args.frozen_part):
            with autocast(enabled=amp):
                y_Y = self.encoder_Y(xInput_Y, alfa=1, h=h, w=w)
                y_UV = self.encoder_UV(xInput_UV, alfa=1, h=h2, w=w2)
        return self._train_forward_to_loss(net_input, n_rate, ft, y_Y, y_UV, criterion, beta, args, amp)

    def _train_forward_to_loss(self, net_input, n_rate, ft, y_Y, y_UV, criterion, beta, args, amp=True):
        #net_input = self._preprocess(x, args)
        #n_rate, ft = self._nrate_n_ft(beta)
        #xInput_Y, xInput_UV = net_input['xInput_Y'], net_input['xInput_UV']
        h, w = net_input['input_shape']['H'], net_input['input_shape']['W']
        h2, w2 = (h+1)>>1, (w+1)>>1
        #with torch.set_grad_enabled(not args.train_entropy_n_synth_parts):
        #    with autocast(enabled=amp):
        #        y_Y = self.encoder_Y(xInput_Y, alfa=1, h=h, w=w)
        #        y_UV = self.encoder_UV(xInput_UV, alfa=1, h=h2, w=w2)

        # quantizer and distribution about `z`
        with autocast(enabled=amp):
            z_Y = self.hyper_encoder_Y(y_Y, h=h, w=w)
            z_UV = self.hyper_encoder_UV(y_UV, h=h2, w=w2)

        z_hat_Y, z_Y_for_lh_calculation = self.addZNoise(z_Y)
        z_hat_UV, z_UV_for_lh_calculation = self.addZNoise(z_UV)

        # distribution about z
        z_likelihoods_Y = self.hyper_entropy_Y(z_Y_for_lh_calculation.to(dtype=torch.float32))
        z_likelihoods_UV = self.hyper_entropy_UV(z_UV_for_lh_calculation.to(dtype=torch.float32))
        # distribution about `y`
        with autocast(enabled=amp):
            psi_Y = self.hyper_decoder_Y(z_hat_Y, h=h, w=w)
            scale_hat_Y = self.hyper_scale_decoder_Y(z_hat_Y, h=h, w=w)
            psi_UV = self.hyper_decoder_UV(z_hat_UV, h=h2, w=w2)
            scale_hat_UV = self.hyper_scale_decoder_UV(z_hat_UV, h=h2, w=w2)

            scale_hat_Y = self._index_to_scale(scale_hat_Y)
            scale_hat_UV = self._index_to_scale(scale_hat_UV)

            scale_hat_Y = self.vr_vec_Y(scale_hat_Y, n_rate, ft)
            scale_hat_UV = self.vr_vec_UV(scale_hat_UV, n_rate, ft)

        # only Y comp do context
        with autocast(enabled=amp):
            gvae_params_y = {"gvae": self.vr_vec_Y.module, "n_rate": n_rate, "ft": ft}
            self.context_Y.module.quantize_func = partial(self.quantize, gvae_params=gvae_params_y)
            y_hat_Y_precise, mean_Y, _, _, _ = self.context_Y(y=y_Y, hyper_params=psi_Y)
        y_hat_Y = y_Y - (y_Y - y_hat_Y_precise).detach()

        mean_UV = Upsample_proc(torch.chunk(psi_UV, chunks=4, dim=1))
        mean_UV = mean_UV[:, :, 0:y_UV.shape[2], 0:y_UV.shape[3]]

        res_Y = y_Y - mean_Y
        res_Y = self.vr_vec_Y(res_Y, n_rate, ft)

        res_UV = y_UV - mean_UV
        res_UV = self.vr_vec_UV(res_UV, n_rate, ft)

        res_Y_for_lh_calculation = self.addResiNoise(res_Y)
        res_UV_for_lh_calculation = self.addResiNoise(res_UV)

        res_UV = self._round_with_detach(res_UV)

        if 'entropy' not in args.frozen_part:
            res_Y_for_lh_calculation = self._round_with_detach(res_Y)
            y_hat_Y = y_hat_Y_precise

        # TODO: convert res_Y_for_lh_calculation to torch.float32
        y_likelihoods_Y = cal_y_likelihoods_decoupled(None,
                                                      None,
                                                      None,
                                                      self.entropy,
                                                      scale_hat_Y.to(dtype=torch.float32),
                                                      res=res_Y_for_lh_calculation)
        y_likelihoods_UV = cal_y_likelihoods_decoupled(None,
                                                       None,
                                                       None,
                                                       self.entropy,
                                                       scale_hat_UV.to(dtype=torch.float32),
                                                       res=res_UV_for_lh_calculation)

        xY_reco, xU_reco, xV_reco = None, None, None
        if args.enable_gvae:
            res_Y = (res_Y.round() - res_Y).detach() + res_Y
        with autocast(enabled=amp):
            if args.skip_thre > 1e-12:
                res_UV, y_likelihoods_UV = self._skip_latent(
                        res_UV,
                        scale_hat_UV,
                        y_likelihoods_UV,
                        skip_thre=args.skip_thre,
                        cube_flag_thre=args.cube_flag_thre
                    )
            if True:
                if args.enable_gvae:
                    y_hat_Y = self.vr_vec_Y.module.dequantize(res_Y, n_rate, ft) + mean_Y
                    y_hat_UV = self.vr_vec_UV.module.dequantize(res_UV, n_rate, ft) + mean_UV
                else:
                    y_hat_Y = self.vr_vec_Y.module.dequantize(y_hat_Y, n_rate, ft)
                    y_hat_UV = self.vr_vec_UV.module.dequantize(res_UV + mean_UV, n_rate, ft)

                xY_reco, xU_reco, xV_reco = self._rec_from_latent_code(y_hat_Y, y_hat_UV, h, w, args)

            reg_loss = self._regulation_loss(criterion, args, amp=amp)
            rd_loss = self._rd_loss(criterion,
                                    y_likelihoods_Y,
                                    y_likelihoods_UV,
                                    z_likelihoods_Y,
                                    z_likelihoods_UV,
                                    xY_reco,
                                    xU_reco,
                                    xV_reco,
                                    net_input,
                                    beta,
                                    args)
            rec_data = dict()
            rec_data['xY'] = xY_reco
            rec_data['xCb'] = xU_reco 
            rec_data['xCr'] = xV_reco
        ans = {'rd_loss': rd_loss, 'reg_loss': reg_loss}
        if args.overfit: 
            ans['rec_data'] = rec_data
        return ans

    def _preprocess(self, x, args, is_train=True):
        input_shape = dict()
        if is_train:
            c, h, w = x.size()[-3:]
            crop_h = random.randint(1, 32) * 2  # multiple of 2 for shuffle
            crop_w = random.randint(1, 32) * 2
            x = x.view(-1, c, h, w)
            if args.bh == 1:
                x = x[:, :, 0:h-crop_h, 0:w-crop_w]
                h, w = x.size()[-2:]
            input_shape['N'], input_shape['C'], input_shape['H'], input_shape['W'] = x.shape

        if self.mem_format == 'channel_last':
            x = x.to(self.device, memory_format=torch.channels_last, non_blocking=True)
        elif self.mem_format == 'channel_first':
            x = x.to(self.device, memory_format=torch.contiguous_format, non_blocking=True)

        # from src.codec.common import Image
        # img = Image.create_from_tensor(x, [0,255], bit_depth=8, color_space='rgb')
            
        x = ColorSpace.rgb_to_yuv(x.div(255)).mul(255)

        if not is_train:

            input_shape['N'], input_shape['C'], input_shape['H'], input_shape['W'] = x.shape
            if input_shape['H'] % 2:
                x = F.pad(x, (0, 0, 0, 1), mode='replicate')
            if input_shape['W'] % 2:
                x = F.pad(x, (0, 1, 0, 0), mode='replicate')
            input_shape['N'], input_shape['C'], input_shape['H'], input_shape['W'] = x.shape
            if input_shape['H'] * input_shape['W'] > args.max_pic_area_in_validation:
                return dict(zip(['skip'], [True]))

        if args.color == '420':  # shuffle the full uv
            xCbCr = x[:, 1:, ::2, ::2] # TODO: remove the xCbCr
            xCbCr_ori = x[:, 1:, :, :]
            xY = x[:, :1, :, :]
            xInput_Y = xY
            xInput_UV = F.pixel_unshuffle(x, 2)
        else:
            #xY = torch.unsqueeze(x[:, 0, :, :], 1)
            xY = img.get_component('a')
            #xCbCr = x[:, 1::, :, :]
            xCbCr = torch.cat((img.get_component('b'),img.get_component('c')), dim=1)
            xInput_Y = xY
            xInput_UV = torch.cat((xY, xCbCr), dim=1)
        out = dict(
            zip(['xCbCr', 'xY', 'xInput_Y', 'xInput_UV', 'input_shape', 'xCbCr_ori'],
                [xCbCr, xY, xInput_Y, xInput_UV, input_shape, xCbCr_ori]))
        return out

    def _nrate_n_ft(self, beta):
        beta_list = self.model_beta_list
        n = 0
        f = 0
        for i, b in enumerate(beta_list):
            if beta >= b:
                n = i
            if i < (len(beta_list) - 1) and (beta >= b) and (beta < beta_list[i + 1]):
                f = (beta - b) / (beta_list[i + 1] - b)
        if beta > max(
                beta_list):  # Extreme case, should not happen in the train, but why not to try
            f = torch.tensor(beta / max(beta_list))
            n = len(beta_list) - 1
        if beta < min(
                beta_list):  # Extreme case, should not happen in the train, but why not to try
            f = -torch.tensor(beta / min(beta_list))
        return n, f

    def _regulation_loss(self, criterion, args, amp):
        reg_loss = torch.tensor(0.0, device=self.device)
        rnab_parameters = []
        for name, param in self.decoder_Y.named_parameters():
            if 'rnab' in name:
                rnab_parameters.append(param)
            if 'CAB' in name:
                rnab_parameters.append(param)
            if 'TAM' in name:
                rnab_parameters.append(param)
        for name, param in self.decoder_UV.named_parameters():
            if 'rnab' in name:
                rnab_parameters.append(param)
            if 'CAB' in name:
                rnab_parameters.append(param)
            if 'TAM' in name:
                rnab_parameters.append(param)
        for name, param in self.encoder_Y.named_parameters():
            if 'rnab' in name:
                rnab_parameters.append(param)
            if 'cab' in name:
                rnab_parameters.append(param)
            if 'TAM' in name:
                rnab_parameters.append(param)
        for name, param in self.encoder_UV.named_parameters():
            if 'rnab' in name:
                rnab_parameters.append(param)
            if 'cab' in name:
                rnab_parameters.append(param)
            if 'TAM' in name:
                rnab_parameters.append(param)
        reg_loss += criterion['reg'](rnab_parameters)
        return reg_loss

    def _distortion_loss(self, loss_func, label, predict, args):
        ls = label.shape
        p = predict.to(dtype=torch.float32).view(self.total_loss_comp, ls[0], ls[1], ls[2], ls[3])
        ans = torch.zeros([1], device=label.device)
        enc_count = self.loss_weights_summ.numel()
        for i in range(self.total_loss_comp):
            lw = self.loss_weights[i]
            loss_val = loss_func(label, p[i])
            tmp = lw * loss_val / self.loss_weights_summ[i % enc_count].item()
            ans += tmp
        distortion_loss = ans / enc_count
        return distortion_loss

    def _rd_loss(self,
                 criterion,
                 y_likelihoods_Y,
                 y_likelihoods_UV,
                 z_likelihoods_Y,
                 z_likelihoods_UV,
                 xY_reco,
                 xU_reco,
                 xV_reco,
                 net_input,
                 beta,
                 args,
                 is_train=True):
        input_shape = net_input['input_shape']
        num_pixel = y_likelihoods_Y.shape[0] * input_shape['H'] * input_shape['W']
        y_rate_loss = (-y_likelihoods_Y.log2().sum() - y_likelihoods_UV.log2().sum()) / num_pixel
        z_rate_loss = (-z_likelihoods_Y.log2().sum() - z_likelihoods_UV.log2().sum()) / num_pixel
        xY, xCbCr = net_input['xY'], net_input['xCbCr_ori']  # full xcbcr for metric caculated
        msssim, mse_ret = 0.0, 0.0
        if False:
            loss = y_rate_loss + z_rate_loss
        else:  # TODO: move distortion calculation to the separate function
            # We need MSE distortion calculation in any case - for logging
            distortion_Y_loss = self._distortion_loss(criterion['mse'], xY, xY_reco, args)
            distortion_Cb_loss = self._distortion_loss(criterion['mse'], xCbCr[:, :1, :, :], xU_reco, args)
            distortion_Cr_loss = self._distortion_loss(criterion['mse'], xCbCr[:, 1:, :, :], xV_reco, args)
            distortion_Y = 10 * np.log10(255 * 255 / (distortion_Y_loss.item() + 1e-10))
            distortion_U = 10 * np.log10(255 * 255 / (distortion_Cb_loss.item() + 1e-10))
            distortion_V = 10 * np.log10(255 * 255 / (distortion_Cr_loss.item() + 1e-10))
            if beta < 0.5: # if is the highest beta
                w_y, w_u, w_v = 0.8, 0.1, 0.1
            else:
                w_y, w_u, w_v = 0.33, 0.33, 0.33
            if args.loss_type == 'mse':
                loss = y_rate_loss + \
                    z_rate_loss + \
                    (args.mse_weight * beta) * (distortion_Y_loss * w_y + distortion_Cb_loss * w_u + distortion_Cr_loss * w_v)
                mse_ret = (distortion_Y_loss * w_y + distortion_Cb_loss * w_u + distortion_Cr_loss * w_v).item()
            elif args.loss_type == 'msssim':
                msssim_Y_loss = self._distortion_loss(criterion['msssim'], xY, xY_reco, args)
                msssim_Cb_loss = self._distortion_loss(criterion['msssim'], xCbCr[:, :1, :, :], xU_reco, args)
                msssim_Cr_loss = self._distortion_loss(criterion['msssim'], xCbCr[:, 1:, :, :], xV_reco, args)
                msssim_Y = (1 - msssim_Y_loss).item()
                msssim_Cb = (1 - msssim_Cb_loss).item()
                msssim_Cr = (1 - msssim_Cr_loss).item()
                msssim = msssim_Y * w_y + msssim_Cb * w_u + msssim_Cr * w_v
                loss = y_rate_loss + z_rate_loss + beta * 1000 * (msssim_Y_loss * w_y + msssim_Cb_loss * w_u + msssim_Cr_loss * w_v)
                mse_ret = (distortion_Y_loss * w_y + distortion_Cb_loss * w_u + distortion_Cr_loss * w_v).item()
            elif args.loss_type == 'mix' and args.msssim_weight is not None:
                mse_Y_loss = self._distortion_loss(criterion['mse'], xY, xY_reco, args)
                mse_Cb_loss = self._distortion_loss(criterion['mse'], xCbCr[:, :1, :, :], xU_reco, args)
                mse_Cr_loss = self._distortion_loss(criterion['mse'], xCbCr[:, 1:, :, :], xV_reco, args)
                msssim_Y_loss = self._distortion_loss(criterion['msssim'], xY, xY_reco, args)
                msssim_Cb_loss = self._distortion_loss(criterion['msssim'], xCbCr[:, :1, :, :], xU_reco, args)
                msssim_Cr_loss = self._distortion_loss(criterion['msssim'], xCbCr[:, 1:, :, :], xV_reco, args)
                msssim_Y = (1 - msssim_Y_loss).item()
                msssim_Cb = (1 - msssim_Cb_loss).item()
                msssim_Cr = (1 - msssim_Cr_loss).item()
                msssim = msssim_Y * w_y + msssim_Cb * w_u + msssim_Cr * w_v
                mse_ret = (mse_Y_loss * w_y + mse_Cb_loss * w_u + mse_Cr_loss * w_v).item()

                factorY, factorCb, factorCr = [float(x) for x in args.loss_factors.split('_')]
                a_ssim = args.msssim_weight
                if args.beta < 0.01: # model_id 0, 0.002
                    distortion_Y_loss =  (1 - a_ssim) * mse_Y_loss * factorY + a_ssim * 1000 * msssim_Y_loss
                    distortion_Cb_loss = (1 - a_ssim) * mse_Cb_loss * factorCb
                    distortion_Cr_loss = (1 - a_ssim) * mse_Cr_loss * factorCr
                elif args.beta < 0.07: # model id 1, 0.012
                    distortion_Y_loss =  (1 - a_ssim) * mse_Y_loss * factorY + a_ssim * 1000 * msssim_Y_loss
                    distortion_Cb_loss = (1 - a_ssim) * mse_Cb_loss * factorCb
                    distortion_Cr_loss = (1 - a_ssim) * mse_Cr_loss * factorCr
                elif args.beta < 0.2: # model id 2, 0.075
                    distortion_Y_loss = (1 - a_ssim) * mse_Y_loss * factorY + a_ssim * 1000 * msssim_Y_loss * factorY
                    distortion_Cb_loss = mse_Cb_loss * factorCb
                    distortion_Cr_loss = mse_Cr_loss * factorCr
                else: # model id 3, 0.5
                    distortion_Y_loss = (1 - a_ssim) * mse_Y_loss * factorY + a_ssim * 1000 * msssim_Y_loss * factorY
                    distortion_Cb_loss = mse_Cb_loss * factorCb
                    distortion_Cr_loss = mse_Cr_loss * factorCr

                loss = y_rate_loss + z_rate_loss + (args.mse_weight * beta) * (distortion_Y_loss + distortion_Cb_loss + distortion_Cr_loss)
            else:
                raise NotImplementedError
        if is_train:
            ret = loss
        else:
            labels = ['Rate', 'Hyper_rate', 'Y_PSNR', 'U_PSNR', 'V_PSNR', 'MSE', 'MSSSIM', 'Loss']
            vals = [
                y_rate_loss, z_rate_loss, distortion_Y, distortion_U, distortion_V, mse_ret,
                msssim,
                loss.item()
            ]
            ret = dict(zip(labels, vals))
        return ret

    def val_forward_to_loss(self, x, criterion, beta, args, amp=False):
        n_rate, ft = self._nrate_n_ft(beta)

        net_input = self._preprocess(x, args, is_train=False)
        if 'skip' in net_input:
            return None
        return self._val_forward_to_loss(net_input, n_rate, ft, criterion, beta, args, amp)

    def _masked_entropy(self, residual_Y_hat, scale_hat_Y, y_likelihoods_Y, residual_UV_hat, scale_hat_UV, y_likelihoods_UV):
        _, masked_y_likelihoods_Y = self._skip_latent(residual_Y_hat, scale_hat_Y, y_likelihoods_Y, skip_thre=0.2, cube_flag_thre=1.0)
        _, masked_y_likelihoods_UV = self._skip_latent(residual_UV_hat, scale_hat_UV, y_likelihoods_UV, skip_thre=0.2, cube_flag_thre=1.0)

        latent_H_Y, latent_W_Y = masked_y_likelihoods_Y.shape[2], masked_y_likelihoods_Y.shape[3]
        entropy_Y = torch.sum(-masked_y_likelihoods_Y.log2(), dim=(0, 2, 3)) / (latent_H_Y * latent_W_Y)

        latent_H_UV, latent_W_UV = masked_y_likelihoods_UV.shape[2], masked_y_likelihoods_UV.shape[3]
        entropy_UV = torch.sum(-masked_y_likelihoods_UV.log2(), dim=(0, 2, 3)) / (latent_H_UV * latent_W_UV)

        self.channel_wise_entropy_Y += entropy_Y.squeeze()
        self.channel_wise_entropy_UV += entropy_UV.squeeze()

    def _val_forward_to_loss(self, net_input, n_rate, ft, criterion, beta, args, amp=False, is_collect=False, stat_Y=None, stat_UV=None, first_pass=False):

        with torch.set_grad_enabled(False):
            # for tiling of Y
            num_downsampling_layers = 3
            alignment_size = tiling.get_alignment_size(num_downsampling_layers)
            tile_manager_Y = TileManager(alignment_size, latent_downscale_factor_y=alignment_size, use_coding_headers=False, signal_tileSignalingType=False)
            tile_manager_Y.numSamplesPerTile = 1024*1024 #/ self.encoder_Y.module.count()
            # tile_manager_Y.numSamplesPerTile = 1000000 # -> 4Ggb
            tile_manager_Y.numSamplesTileOverlap = 48
            img_height_Y, img_width_Y = net_input['xInput_Y'].shape[2:]
            latent_height_Y = math.ceil(img_height_Y / alignment_size)
            latent_width_Y = math.ceil(img_width_Y / alignment_size)
            latent_shape_Y = (self.encoder_Y.module.count(), args.N, latent_height_Y, latent_width_Y) # channels not used for tiling
            
            latent_y_height = math.ceil(img_height_Y / alignment_size)
            latent_y_width = math.ceil(img_width_Y / alignment_size)
            latent_y_shape = (1, args.N, latent_y_height, latent_y_width)
            latent_psi_height = math.ceil(img_height_Y / (alignment_size * 2 ))
            latent_psi_width = math.ceil(img_width_Y / (alignment_size * 2 ))
            latent_psi_shape = (1, args.N*4, latent_psi_height, latent_psi_width)
            latent_z_height = math.ceil(img_height_Y / (alignment_size * 4 ))
            latent_z_width = math.ceil(img_width_Y / (alignment_size * 4 ))
            latent_z_shape = (1, args.N*4, latent_z_height, latent_z_width)
            tile_manager_Y.setup_tiles_enc(net_input['xInput_Y'].shape, latent_y_shape, latent_psi_shape, latent_z_shape)


            # for tiling of UV
            num_downsampling_layers = 2
            num_upsampling_layers = 3
            alignment_size = tiling.get_alignment_size(num_downsampling_layers)
            alignment_size_dec = tiling.get_alignment_size(num_upsampling_layers)
            tile_manager_UV = TileManager(alignment_size, latent_downscale_factor_y=alignment_size, use_coding_headers=False, signal_tileSignalingType=False)
            tile_manager_UV.numSamplesPerTile = 1179648 / self.encoder_UV.module.count()
            # tile_manager_UV.numSamplesPerTile = 250000 # -> 4Ggb
            tile_manager_UV.numSamplesTileOverlap = 48
            img_height_UV, img_width_UV = net_input['xInput_UV'].shape[2:]
            latent_height_UV = math.ceil(img_height_UV / alignment_size)
            latent_width_UV = math.ceil(img_width_UV / alignment_size)
            latent_shape_UV = (self.encoder_UV.module.count(), args.N_UV, latent_height_UV, latent_width_UV) # channels not used for tiling

            latent_y_height = math.ceil(img_height_UV / alignment_size)
            latent_y_width = math.ceil(img_width_UV / alignment_size)
            latent_y_shape = (1, args.N_UV, latent_y_height, latent_y_width)
            latent_psi_height = math.ceil(img_height_UV / (alignment_size * 2 ))
            latent_psi_width = math.ceil(img_width_UV / (alignment_size * 2 ))
            latent_psi_shape = (1, args.N_UV*4, latent_psi_height, latent_psi_width)
            latent_z_height = math.ceil(img_height_UV / (alignment_size * 4 ))
            latent_z_width = math.ceil(img_width_UV / (alignment_size * 4 ))
            latent_z_shape = (1, args.N_UV*4, latent_z_height, latent_z_width)
            tile_manager_UV.setup_tiles_enc(net_input['xInput_UV'].shape, latent_y_shape, latent_psi_shape, latent_z_shape)


            # for tiling of UV decoder 444
            tile_manager_UV_dec = TileManager(alignment_size_dec, latent_downscale_factor_y=alignment_size_dec, use_coding_headers=False, signal_tileSignalingType=False)
            tile_manager_UV_dec.numSamplesPerTile = 1179648 / self.decoder_UV.module.count()
            # tile_manager_UV.numSamplesPerTile = 250000 # -> 4Ggb
            tile_manager_UV_dec.numSamplesTileOverlap = 48
            img_height_UV, img_width_UV = net_input['xInput_UV'].shape[2:]
            img_height_UV, img_width_UV = img_height_UV*2, img_width_UV*2
            img_shape = (1, net_input['xInput_UV'].shape[1], img_height_UV, img_width_UV)
            tile_manager_UV_dec.setup_tiles_enc(img_shape, latent_y_shape, latent_psi_shape, latent_z_shape)

            #################### encoder Y ###########################
            xInput_Y = net_input['xInput_Y']
            n_i, c_i, h_i, w_i = xInput_Y.shape
            h_i2, w_i2 = (h_i+1)>>1, (w_i+1)>>1
            if tile_manager_Y.is_enabled():
                y_Y = torch.empty(latent_shape_Y, device=xInput_Y.device)

                for image_tile, latent_tile in zip(tile_manager_Y.image_tiles,
                                                   tile_manager_Y.latent_tiles):
                    image_tile_data = tiling.get_data(xInput_Y, image_tile)
                    if not is_collect:
                        y_tile = self.encoder_Y(image_tile_data, alfa=1, h=image_tile.size.height, w=image_tile.size.width)
                    else:
                        y_tile, stat_Y = self.encoder_Y(image_tile_data, alfa=1, h=image_tile.size.height, w=image_tile.size.width, is_collect=True, stat=stat_Y, first_pass=first_pass)

                    # assign tile feature, but leave out features at boundaries by not using
                    # part of overlapped regions
                    assigned_tile, assigned_tile_rel_to_overlap = tile_manager_Y.get_core_of_overlapping_latent_tile(
                        latent_tile, image_tile, None)
                    assigned_data = tiling.get_data(y_tile, assigned_tile_rel_to_overlap)
                    tiling.assign_data(y_Y, assigned_tile, assigned_data)

            else:
                if not is_collect:
                    y_Y = self.encoder_Y(xInput_Y, alfa=1, h=h_i, w=w_i)
                else:
                    y_Y, stat_Y = self.encoder_Y(xInput_Y, alfa=1, h=h_i, w=w_i, is_collect=True, stat=stat_Y, first_pass=first_pass)

            #################### encoder UV ###########################
            xInput_UV = net_input['xInput_UV']
            if tile_manager_UV.is_enabled():
                y_UV = torch.empty(latent_shape_UV, device=xInput_UV.device)

                for image_tile, latent_tile in zip(tile_manager_UV.image_tiles,
                                                   tile_manager_UV.latent_tiles):
                    image_tile_data = tiling.get_data(xInput_UV, image_tile)
                    if not is_collect:
                        y_tile = self.encoder_UV(image_tile_data, alfa=1, h=image_tile.size.height, w=image_tile.size.width)
                    else:
                        y_tile, stat_UV = self.encoder_UV(image_tile_data, alfa=1, h=image_tile.size.height, w=image_tile.size.width, is_collect=True, stat=stat_UV, first_pass=first_pass)

                    # assign tile feature, but leave out features at boundaries by not using
                    # part of overlapped regions
                    assigned_tile, assigned_tile_rel_to_overlap = tile_manager_UV.get_core_of_overlapping_latent_tile(
                        latent_tile, image_tile, None)
                    assigned_data = tiling.get_data(y_tile, assigned_tile_rel_to_overlap)
                    tiling.assign_data(y_UV, assigned_tile, assigned_data)

            else:
                if args.color == '420':
                    if not is_collect:
                        y_UV = self.encoder_UV(xInput_UV,
                                               alfa=1,
                                               h=h_i2,
                                               w=w_i2)
                    else:
                        y_UV, stat_UV = self.encoder_UV(xInput_UV,
                                                        alfa=1,
                                                        h=h_i2,
                                                        w=w_i2, 
                                                        is_collect=True, stat=stat_UV, first_pass=first_pass)
                else:
                    if not is_collect:
                        y_UV = self.encoder_UV(xInput_UV, alfa=1, h=h_i, w=w_i)
                    else:
                        y_UV, stat_UV = self.encoder_UV(xInput_UV, alfa=1, h=h_i, w=w_i, is_collect=True, stat=stat_UV, first_pass=first_pass)

            #y_UV = self.func_encoder_UV(y_UV, n_rate, ft)
 
            #################### hyper encoder/decoder Y, UV ###########################
            if not is_collect:
                z_Y = self.hyper_encoder_Y(y_Y, h=h_i, w=w_i)
            else:
                z1, z2, z3, z4, z_Y = self.hyper_encoder_Y(y_Y, h=h_i, w=w_i, is_collect=True)
                getStat(z1, stat_Y['HE1'], first_pass)
                getStat(z2, stat_Y['HE2'], first_pass)
                getStat(z3, stat_Y['HE3'], first_pass)
                getStat(z4, stat_Y['HE4'], first_pass)
                getStat(z_Y, stat_Y['HE5'], first_pass)
            if args.color == '420':
                if not is_collect:
                    z_UV = self.hyper_encoder_UV(y_UV, h=h_i2, w=w_i2)
                else:
                    z1, z2, z3, z4, z_UV = self.hyper_encoder_UV(y_UV, h=h_i2, w=w_i2, is_collect=True)
                    getStat(z1, stat_UV['HE1'], first_pass)
                    getStat(z2, stat_UV['HE2'], first_pass)
                    getStat(z3, stat_UV['HE3'], first_pass)
                    getStat(z4, stat_UV['HE4'], first_pass)
                    getStat(z_UV, stat_UV['HE5'], first_pass)
            else:
                if not is_collect:
                    z_UV = self.hyper_encoder_UV(y_UV, h=h_i, w=w_i)
                else:
                    z1, z2, z3, z4, z_UV = self.hyper_encoder_UV(y_UV, h=h_i, w=w_i, is_collect=True)
                    getStat(z1, stat_UV['HE1'], first_pass)
                    getStat(z2, stat_UV['HE2'], first_pass)
                    getStat(z3, stat_UV['HE3'], first_pass)
                    getStat(z4, stat_UV['HE4'], first_pass)
                    getStat(z_UV, stat_UV['HE5'], first_pass)

            z_hat_Y = z_Y.round()
            z_hat_UV = z_UV.round()

            # distribution about z
            z_likelihoods_Y = self.hyper_entropy_Y(z_hat_Y.to(dtype=torch.float32))
            z_likelihoods_UV = self.hyper_entropy_UV(z_hat_UV.to(dtype=torch.float32))
            # distribution about `y`
            psi_Y = self.hyper_decoder_Y(z_hat_Y, h=h_i, w=w_i)
            scale_hat_Y = self.hyper_scale_decoder_Y(z_hat_Y, h=h_i, w=w_i)
            if args.color == '420':
                psi_UV = self.hyper_decoder_UV(z_hat_UV,
                                               h=h_i2,
                                               w=w_i2)
                scale_hat_UV = self.hyper_scale_decoder_UV(z_hat_UV,
                                                           h=h_i2,
                                                           w=w_i2)
            else:
                psi_UV = self.hyper_decoder_UV(z_hat_UV, h=h_i, w=w_i)
                scale_hat_UV = self.hyper_scale_decoder_UV(z_hat_UV, h=h_i, w=w_i)
   
            scale_hat_Y = self._index_to_scale(scale_hat_Y)
            scale_hat_UV = self._index_to_scale(scale_hat_UV)

            scale_hat_Y = self.vr_vec_Y(scale_hat_Y, n_rate, ft)
            scale_hat_UV = self.vr_vec_UV(scale_hat_UV, n_rate, ft)

            gvae_params_y = {"gvae": self.vr_vec_Y.module, "n_rate": n_rate, "ft": ft}
            self.context_Y.module.quantize_func = partial(self.quantize, gvae_params=gvae_params_y)
            y_hat_Y, mean_Y, _, _, _ = self.context_Y(y=y_Y, hyper_params=psi_Y)

            # gvae_params_uv = {"gvae": self.vr_vec_UV.module, "n_rate": n_rate, "ft": ft}
            # self.context_UV.module.quantize_func = partial(self.quantize, gvae_params=gvae_params_uv)
            # y_hat_UV, mean_UV, _, _ = self.context_UV(y=y_UV, hyper_params=psi_UV)
            mean_UV = Upsample_proc(torch.chunk(psi_UV,chunks=4,dim=1))
            mean_UV = mean_UV[:,:,0:y_UV.shape[2], 0:y_UV.shape[3]]

            residual_Y_hat  = self.vr_vec_Y(y_Y - mean_Y , n_rate, ft).round()
            residual_UV_hat = self.vr_vec_UV(y_UV - mean_UV, n_rate, ft).round()

            y_hat_UV = mean_UV + residual_UV_hat


            y_likelihoods_Y = cal_y_likelihoods_decoupled(None,
                                                          None,
                                                          None,
                                                          self.entropy,
                                                          scale_hat_Y,
                                                          y=None,
                                                          res=residual_Y_hat)
            y_likelihoods_UV = cal_y_likelihoods_decoupled(None,
                                                           None,
                                                           None,
                                                           self.entropy,
                                                           scale_hat_UV,
                                                           y=None,
                                                           res=residual_UV_hat)

            if self.args.cal_entropy_on_val:
                self._masked_entropy(residual_Y_hat, scale_hat_Y, y_likelihoods_Y, residual_UV_hat, scale_hat_UV, y_likelihoods_UV)
            with autocast(enabled=amp):
                #################### decoder Y ###########################
                xY_reco, xU_reco, xV_reco = self._tiled_rec_from_latent_code(net_input, y_hat_Y, y_hat_UV, h_i, w_i, tile_manager_Y, tile_manager_UV_dec, args)

            # reg_loss = self._regulation_loss(criterion, args, amp=False)
            val_infos = self._rd_loss(criterion,
                                      y_likelihoods_Y,
                                      y_likelihoods_UV,
                                      z_likelihoods_Y,
                                      z_likelihoods_UV,
                                      xY_reco,
                                      xU_reco,
                                      xV_reco,
                                      net_input,
                                      beta,
                                      args,
                                      is_train=False)
            if not is_collect:
                if args.overfit:
                    rec_data = dict()
                    rec_data['xY'] = xY_reco
                    rec_data['xCb'] = xU_reco 
                    rec_data['xCr'] = xV_reco
                    val_infos['rec_data'] = rec_data
                return val_infos
            else:
                return stat_Y, stat_UV

    def data_collection_forward(self, x, criterion, beta, args, stat_Y, stat_UV, first_pass=True, amp=False):
        n_rate, ft = self._nrate_n_ft(beta)

        net_input = self._preprocess(x, args, is_train=True)
        if 'skip' in net_input:
            return None
        return self._val_forward_to_loss(net_input, n_rate, ft, criterion, beta, args, amp, is_collect=True, stat_Y=stat_Y, stat_UV=stat_UV, first_pass=first_pass)
