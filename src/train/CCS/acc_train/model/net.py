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
import os
import random

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
from src.codec.components.contexts.utils import ContextUtils
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
        HyperScaleDecoderFactory,
        QuantModule
    )
from src.codec.coding_tools.tiling import TileManager
from src.codec.common import tiling
from typing import Tuple, List, Union

__all__ = ['Net']


def _is_main_process() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_beta_list(args):
    beta_list = [float(x) for x in args.beta_list.split(',')]
    if _is_main_process():
        print('beta list for current train =', beta_list)
    model_beta_list = [
        0.0005, 0.001, 0.002, 0.004, 0.007, 0.01, 0.012, 0.015, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5, 0.75, 1.0, 2.0, 3.0
    ]
    return beta_list, model_beta_list


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
    
    def forward(self, img: torch.Tensor, h:int, w:int, *args, **kwargs) -> torch.Tensor:
        ans = list()
        for coder in self.coders.values():
            ans.append(coder(img, h=h, w=w, *args, **kwargs)[..., :h, :w])
        return torch.cat(ans, dim=0)

class Net(nn.Module):
    """AI Codec Net"""
    def __init__(self, args):
        super(Net, self).__init__()
        N, N_UV, N_Y, _ = args.N, args.N_UV, args.N_Y, 1
        self.args = args
        self.mem_format = 'channel_first'
        init_device = torch.device(getattr(args, 'local_rank', 0) if torch.cuda.is_available() else 'cpu')
        self.beta_list, self.model_beta_list = get_beta_list(args)
        self.vae_encoder_type_list = args.vae_encoder_type_list
        self.vae_decoder_type_list = args.vae_decoder_type_list

        self.sigma_min = args.sigma_quant_min
        self.sigma_max = args.sigma_quant_max
        self.sigma_level = args.sigma_quant_level
        self.log_k = GMProbModel.scale_log_step(self.sigma_min, self.sigma_max, self.sigma_level)

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

        self.register_buffer('channel_wise_entropy_Y', torch.zeros(N, dtype=torch.float32))
        self.register_buffer('channel_wise_entropy_UV', torch.zeros(N_UV, dtype=torch.float32))

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
        # Not persistent: it is derived from the command line and must not appear in checkpoints.
        self.register_buffer('loss_weights', self._loss_weight_parse(args.loss_weights), persistent=False)
        self._to_device(init_device)
        enc_count = len(args.vae_encoder_type_list)
        self.loss_weights_summ = torch.zeros([enc_count])
        for enc_num in range(enc_count):
            self.loss_weights_summ[enc_num] = self.loss_weights[enc_num::enc_count].sum()
            
        self._register_load_state_dict_pre_hook(self._load_state_dict_hook)

    @property
    def device(self) -> torch.device:
        """Device of the model. It follows ``.to()``/``.cuda()`` calls (e.g. done by Lightning)."""
        return self.channel_wise_entropy_Y.device

    def zero_channel_wise_entropy(self):
        self.channel_wise_entropy_Y.fill_(0)
        self.channel_wise_entropy_UV.fill_(0)
        
    def _loss_weight_parse(self, weights_str: str) -> torch.Tensor:
        
        weights_arr = [[float(y) for y in x.split(',')] for x in weights_str.split(';')]
        
        return torch.tensor(weights_arr, dtype=torch.float).flatten()
        

            
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

        # Vrq quantize the residual
        data_q = vrq_func(data, n_rate, ft) if gvae_params is not None else data

        # skip block and mask padding boundary, clip and round (as the codec does)
        resi_rq = QuantModule.round_masked_ste(data_q, tool_params["mask2"], torch.int32)

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

    def _to_device(self, device):
        self.to(device)

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
        # activation clipping thresholds of the encoders are not used by the training
        checkpoint = self.remove_keys(checkpoint, [k for k in checkpoint if k.endswith('clip_thres')])
        if _is_distributed():
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
        """Save the checkpoint in the legacy (``*.pth``) format.

        The file holds ``epoch`` (next epoch to train), ``best_loss``, ``optimizer`` (its state dict)
        and the model state dict without ``module.`` prefixes, which is the format the codec, the
        automatic testing (``copy_model_for_test``/``split_cp``) and ``--resume`` expect.
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
            if k.endswith('clip_thres'):  # not produced by the training
                del s[k]
                continue
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

    @staticmethod
    def _gloo_ddp(module):
        """DDP on CPU: gloo cannot broadcast bool tensors, the bool buffers (constant flags such as
        ``is_quantized``) are excluded from the synchronisation."""
        bool_buffers = [name for name, buf in module.named_buffers() if buf.dtype == torch.bool]
        if bool_buffers:
            DDP._set_params_and_buffers_to_ignore_for_model(module, bool_buffers)
        return DDP(module)

    def DDP(self, local_rank):
        """Wrap every sub-network into its own ``DistributedDataParallel``.

        Every sub-network is synchronised separately (the training step may run the forward pass
        several times, e.g. the fp32 re-run after an AMP overflow). Without an initialised process
        group, or with a single process, ``DummyDDP`` is used: it only provides the ``.module``
        attribute the forward code relies on.
        """
        if _is_distributed() and dist.get_world_size() > 1:
            # with a single process SyncBatchNorm is the same as BatchNorm (and needs CUDA in torch 1.10)
            self._replace_batchnorm()
            if torch.cuda.is_available():
                ddp_obj = partial(DDP, device_ids=[local_rank], output_device=local_rank)
            else:
                ddp_obj = self._gloo_ddp
        else:
            ddp_obj = DummyDDP
        self.encoder_Y = ddp_obj(self.encoder_Y)
        self.encoder_UV = ddp_obj(self.encoder_UV)
        self.vr_vec_Y = ddp_obj(self.vr_vec_Y)
        self.vr_vec_UV = ddp_obj(self.vr_vec_UV)
        self.decoder_Y = ddp_obj(self.decoder_Y)
        self.decoder_UV = ddp_obj(self.decoder_UV)
        self.hyper_encoder_Y = ddp_obj(self.hyper_encoder_Y)
        self.hyper_encoder_UV = ddp_obj(self.hyper_encoder_UV)
        self.hyper_decoder_Y = ddp_obj(self.hyper_decoder_Y)
        self.hyper_decoder_UV = ddp_obj(self.hyper_decoder_UV)
        self.hyper_scale_decoder_Y = ddp_obj(self.hyper_scale_decoder_Y)
        self.hyper_scale_decoder_UV = ddp_obj(self.hyper_scale_decoder_UV)
        self.hyper_entropy_Y = ddp_obj(self.hyper_entropy_Y)
        self.hyper_entropy_UV = ddp_obj(self.hyper_entropy_UV)
        self.context_Y = ddp_obj(self.context_Y)
        #self.context_UV = DDP(self.context_UV, [local_rank], local_rank)

    def parameter_groups(self):
        """Parameters split into the groups used by the optimizer (and by ``--frozen_part``).

        Returns:
            tuple: (analysis, synthesis, entropy part, gain unit) lists of param-group dicts.
        """
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

    def _gen_skip_cubeflag(self, res_hat, mask_map, cube_flag_thre):
        res_hat_skip = res_hat.clone()
        res_hat_skip[mask_map] = 0
        diff_yhat = (res_hat_skip - res_hat).abs()
        C, H, W = diff_yhat.shape[1:]
        cube_size = 8
        cubeflag = ContextUtils.cube_flags_from_diff(diff_yhat, C, cube_size, cube_flag_thre)
        return ContextUtils.repeat_cube_flags(cubeflag, C, cube_size)[:, :C, :H, :W]

    def _skip_latent(self, res_hat, scale, likelihoods, skip_thre=0.2, cube_flag_thre=1):
        mask_map = (scale <= skip_thre)
        cubeflag = self._gen_skip_cubeflag(res_hat, mask_map, cube_flag_thre)
        mask_map = mask_map & cubeflag
        res_hat[mask_map] = 0
        likelihoods[mask_map] = 1.0
        return res_hat, likelihoods

    # ##################################################################################################################
    #  tiling (validation of big pictures); the tiles are managed by the TileManager of the codec
    # ##################################################################################################################
    NUM_SAMPLES_PER_TILE_Y = 1024 * 1024
    NUM_SAMPLES_PER_TILE_UV = 1179648
    NUM_SAMPLES_TILE_OVERLAP = 48

    def _create_tile_manager(self, alignment_size, num_samples_per_tile, image_shape, latent_shapes) -> TileManager:
        tile_manager = TileManager(alignment_size, latent_downscale_factor_y=alignment_size,
                                   use_coding_headers=False, signal_tileSignalingType=False)
        tile_manager.numSamplesPerTile = num_samples_per_tile
        tile_manager.numSamplesTileOverlap = self.NUM_SAMPLES_TILE_OVERLAP
        tile_manager.setup_tiles_enc(image_shape, *latent_shapes)
        return tile_manager

    def _setup_tile_managers(self, xInput_Y, xInput_UV, args):
        """Tile managers of a picture: luma (analysis and synthesis), chroma analysis and chroma synthesis.

        Returns also the shapes of the latents of all encoders (luma, chroma).
        """
        # luma
        alignment_size = tiling.get_alignment_size(num_downsampling_layers=3)
        latent_shapes_Y = tiling.get_latent_shapes(*xInput_Y.shape[2:], alignment_size, args.N)
        tile_manager_Y = self._create_tile_manager(alignment_size, self.NUM_SAMPLES_PER_TILE_Y,
                                                   xInput_Y.shape, latent_shapes_Y)
        # chroma, analysis
        alignment_size = tiling.get_alignment_size(num_downsampling_layers=2)
        latent_shapes_UV = tiling.get_latent_shapes(*xInput_UV.shape[2:], alignment_size, args.N_UV)
        tile_manager_UV = self._create_tile_manager(
            alignment_size, self.NUM_SAMPLES_PER_TILE_UV / self.encoder_UV.module.count(), xInput_UV.shape, latent_shapes_UV)
        # chroma, synthesis (its output is 4:4:4)
        alignment_size_dec = tiling.get_alignment_size(num_downsampling_layers=3)
        img_shape = (1, xInput_UV.shape[1], xInput_UV.shape[2] * 2, xInput_UV.shape[3] * 2)
        tile_manager_UV_dec = self._create_tile_manager(
            alignment_size_dec, self.NUM_SAMPLES_PER_TILE_UV / self.decoder_UV.module.count(), img_shape, latent_shapes_UV)

        latent_shape_Y = (self.encoder_Y.module.count(), *latent_shapes_Y[0][1:])
        latent_shape_UV = (self.encoder_UV.module.count(), *latent_shapes_UV[0][1:])
        return tile_manager_Y, tile_manager_UV, tile_manager_UV_dec, latent_shape_Y, latent_shape_UV

    @staticmethod
    def _analysis(encoder, x, h, w, tile_manager=None, latent_shape=None):
        """Analysis transform of the whole picture or tile by tile (if tiles are enabled in the tile manager)."""
        if tile_manager is not None and tile_manager.is_enabled():
            return tile_manager.analysis_by_tiles(x, partial(encoder, alfa=1), latent_shape)
        return encoder(x, alfa=1, h=h, w=w)

    @staticmethod
    def _synthesis(decoder, y_hat, h, w, tile_manager=None, image_shape=None):
        """Synthesis transform of the whole picture or tile by tile (if tiles are enabled in the tile manager)."""
        if tile_manager is not None and tile_manager.is_enabled():
            return tile_manager.synthesis_by_tiles(y_hat, decoder, image_shape)
        return decoder(y_hat, h=h, w=w)

    def _rec_from_latent_code(self, y_hat_Y, y_hat_UV, h, w, net_input=None, tile_manager_Y=None, tile_manager_UV=None):
        """Reconstruct the picture; the tile managers (and `net_input`) are given for a tiled reconstruction."""
        # Keep the order of the operations (luma synthesis, concatenation, chroma synthesis): it defines the
        # order the gradients of `y_hat_Y` are accumulated in, i.e. the exact values of the gradients.
        image_shape = None
        if tile_manager_Y is not None:
            image_shape = [y_hat_Y.shape[0] * self.decoder_Y.module.count()] + list(net_input['xInput_Y'].shape[1:])
        xY_reco = self._synthesis(self.decoder_Y, y_hat_Y, h, w, tile_manager_Y, image_shape)

        y_hat_UV = torch.cat((y_hat_Y, y_hat_UV), dim=1)
        if tile_manager_UV is not None:
            xInput_UV = net_input['xInput_UV']
            image_shape = [y_hat_UV.shape[0] * self.decoder_UV.module.count(), 2,
                           2 * xInput_UV.shape[2], 2 * xInput_UV.shape[3]]
        x_reco_UV = self._synthesis(self.decoder_UV, y_hat_UV, h, w, tile_manager_UV, image_shape)
        xU_reco, xV_reco = x_reco_UV.chunk(2, dim=1)
        return xY_reco, xU_reco, xV_reco

    # ##################################################################################################################
    #  parts of the forward pass shared by the training and the validation
    # ##################################################################################################################
    def _hyper_path(self, y_Y, y_UV, size_Y, size_UV, n_rate, ft, amp, add_noise):
        """Hyper analysis, quantization of z and hyper synthesis.

        Args:
            size_Y, size_UV: (height, width) of the inputs of the analysis transforms
            add_noise (bool): training mode - the likelihoods of z are estimated with the uniform noise
                and the gradient is passed through the rounding; otherwise z is just rounded

        Returns:
            likelihoods of z (Y, UV), outputs of the hyper decoders (Y, UV) and scales of the residuals (Y, UV)
        """
        (h, w), (h2, w2) = size_Y, size_UV
        with autocast(enabled=amp):
            z_Y = self.hyper_encoder_Y(y_Y, h=h, w=w)
            z_UV = self.hyper_encoder_UV(y_UV, h=h2, w=w2)

        if add_noise:
            z_hat_Y, z_Y_for_lh_calculation = QuantModule.round_ste(z_Y), QuantModule.noise_ste(z_Y)
            z_hat_UV, z_UV_for_lh_calculation = QuantModule.round_ste(z_UV), QuantModule.noise_ste(z_UV)
        else:
            z_hat_Y = z_Y_for_lh_calculation = z_Y.round()
            z_hat_UV = z_UV_for_lh_calculation = z_UV.round()

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
        return z_likelihoods_Y, z_likelihoods_UV, psi_Y, psi_UV, scale_hat_Y, scale_hat_UV

    def _context_path(self, y_Y, y_UV, psi_Y, psi_UV, n_rate, ft, amp):
        """Prediction of the latents: the context model for luma, the output of the hyper decoder for chroma.

        Returns:
            reconstructed luma latent (in the domain of the gain unit), means of the luma and chroma latents
        """
        with autocast(enabled=amp):
            gvae_params_y = {"gvae": self.vr_vec_Y.module, "n_rate": n_rate, "ft": ft}
            self.context_Y.module.quantize_func = partial(self.quantize, gvae_params=gvae_params_y)
            y_hat_Y, mean_Y, _, _, _ = self.context_Y(y=y_Y, hyper_params=psi_Y)
        mean_UV = ContextUtils.hyper_to_mean(psi_UV, y_UV.shape[2], y_UV.shape[3])
        return y_hat_Y, mean_Y, mean_UV

    def _y_likelihoods(self, scale_hat, res):
        return cal_y_likelihoods_decoupled(None, None, None, self.entropy, scale_hat.to(dtype=torch.float32), res=res)

    def _index_to_scale(self, index):
        return GMProbModel.index_to_scale(index, self.sigma_min, self.sigma_max, self.sigma_level)

    def train_forward_to_loss(self, x, criterion, beta, args, amp=True):
        net_input = self._preprocess(x, args)
        n_rate, ft = self._nrate_n_ft(beta)
        xInput_Y, xInput_UV = net_input['xInput_Y'], net_input['xInput_UV']
        h, w = net_input['input_shape']['H'], net_input['input_shape']['W']
        h2, w2 = (h+1)>>1, (w+1)>>1
        with torch.set_grad_enabled('analysis' not in args.frozen_part):
            with autocast(enabled=amp):
                y_Y = self._analysis(self.encoder_Y, xInput_Y, h, w)
                y_UV = self._analysis(self.encoder_UV, xInput_UV, h2, w2)

        # quantizer and distributions about `z` and `y`
        z_likelihoods_Y, z_likelihoods_UV, psi_Y, psi_UV, scale_hat_Y, scale_hat_UV = self._hyper_path(
            y_Y, y_UV, (h, w), (h2, w2), n_rate, ft, amp, add_noise=True)

        # only Y comp do context
        y_hat_Y_precise, mean_Y, mean_UV = self._context_path(y_Y, y_UV, psi_Y, psi_UV, n_rate, ft, amp)
        y_hat_Y = y_Y - (y_Y - y_hat_Y_precise).detach()

        res_Y = self.vr_vec_Y(y_Y - mean_Y, n_rate, ft)
        res_UV = self.vr_vec_UV(y_UV - mean_UV, n_rate, ft)

        res_Y_for_lh_calculation = QuantModule.noise_ste(res_Y)
        res_UV_for_lh_calculation = QuantModule.noise_ste(res_UV)

        res_UV = QuantModule.round_ste(res_UV)

        if 'entropy' not in args.frozen_part:
            res_Y_for_lh_calculation = QuantModule.round_ste(res_Y)
            y_hat_Y = y_hat_Y_precise

        # TODO: convert res_Y_for_lh_calculation to torch.float32
        y_likelihoods_Y = self._y_likelihoods(scale_hat_Y, res_Y_for_lh_calculation)
        y_likelihoods_UV = self._y_likelihoods(scale_hat_UV, res_UV_for_lh_calculation)

        if args.enable_gvae:
            res_Y = QuantModule.round_ste(res_Y)
        with autocast(enabled=amp):
            if args.skip_thre > 1e-12:
                res_UV, y_likelihoods_UV = self._skip_latent(
                        res_UV,
                        scale_hat_UV,
                        y_likelihoods_UV,
                        skip_thre=args.skip_thre,
                        cube_flag_thre=args.cube_flag_thre
                    )
            if args.enable_gvae:
                y_hat_Y = self.vr_vec_Y.module.dequantize(res_Y, n_rate, ft) + mean_Y
                y_hat_UV = self.vr_vec_UV.module.dequantize(res_UV, n_rate, ft) + mean_UV
            else:
                y_hat_Y = self.vr_vec_Y.module.dequantize(y_hat_Y, n_rate, ft)
                y_hat_UV = self.vr_vec_UV.module.dequantize(res_UV + mean_UV, n_rate, ft)

            xY_reco, xU_reco, xV_reco = self._rec_from_latent_code(y_hat_Y, y_hat_UV, h, w)

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
        ans = {'rd_loss': rd_loss, 'reg_loss': reg_loss}
        if args.overfit:
            ans['rec_data'] = {'xY': xY_reco, 'xCb': xU_reco, 'xCr': xV_reco}
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

        with torch.set_grad_enabled(False):
            xInput_Y, xInput_UV = net_input['xInput_Y'], net_input['xInput_UV']
            h_i, w_i = xInput_Y.shape[2:]
            h_uv, w_uv = ((h_i+1)>>1, (w_i+1)>>1) if args.color == '420' else (h_i, w_i)

            # big pictures are processed by tiles
            tile_manager_Y, tile_manager_UV, tile_manager_UV_dec, latent_shape_Y, latent_shape_UV = \
                self._setup_tile_managers(xInput_Y, xInput_UV, args)

            y_Y = self._analysis(self.encoder_Y, xInput_Y, h_i, w_i, tile_manager_Y, latent_shape_Y)
            y_UV = self._analysis(self.encoder_UV, xInput_UV, h_uv, w_uv, tile_manager_UV, latent_shape_UV)

            # hyper encoder/decoder Y, UV
            z_likelihoods_Y, z_likelihoods_UV, psi_Y, psi_UV, scale_hat_Y, scale_hat_UV = self._hyper_path(
                y_Y, y_UV, (h_i, w_i), (h_uv, w_uv), n_rate, ft, amp=False, add_noise=False)

            y_hat_Y, mean_Y, mean_UV = self._context_path(y_Y, y_UV, psi_Y, psi_UV, n_rate, ft, amp=False)

            residual_Y_hat  = self.vr_vec_Y(y_Y - mean_Y , n_rate, ft).round()
            residual_UV_hat = self.vr_vec_UV(y_UV - mean_UV, n_rate, ft).round()

            y_hat_UV = mean_UV + residual_UV_hat

            y_likelihoods_Y = self._y_likelihoods(scale_hat_Y, residual_Y_hat)
            y_likelihoods_UV = self._y_likelihoods(scale_hat_UV, residual_UV_hat)

            if self.args.cal_entropy_on_val:
                self._masked_entropy(residual_Y_hat, scale_hat_Y, y_likelihoods_Y, residual_UV_hat, scale_hat_UV, y_likelihoods_UV)
            with autocast(enabled=amp):
                xY_reco, xU_reco, xV_reco = self._rec_from_latent_code(
                    y_hat_Y, y_hat_UV, h_i, w_i, net_input, tile_manager_Y, tile_manager_UV_dec)

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
            if args.overfit:
                val_infos['rec_data'] = {'xY': xY_reco, 'xCb': xU_reco, 'xCr': xV_reco}
            return val_infos

    def _masked_entropy(self, residual_Y_hat, scale_hat_Y, y_likelihoods_Y, residual_UV_hat, scale_hat_UV, y_likelihoods_UV):
        _, masked_y_likelihoods_Y = self._skip_latent(residual_Y_hat, scale_hat_Y, y_likelihoods_Y, skip_thre=0.2, cube_flag_thre=1.0)
        _, masked_y_likelihoods_UV = self._skip_latent(residual_UV_hat, scale_hat_UV, y_likelihoods_UV, skip_thre=0.2, cube_flag_thre=1.0)

        latent_H_Y, latent_W_Y = masked_y_likelihoods_Y.shape[2], masked_y_likelihoods_Y.shape[3]
        entropy_Y = torch.sum(-masked_y_likelihoods_Y.log2(), dim=(0, 2, 3)) / (latent_H_Y * latent_W_Y)

        latent_H_UV, latent_W_UV = masked_y_likelihoods_UV.shape[2], masked_y_likelihoods_UV.shape[3]
        entropy_UV = torch.sum(-masked_y_likelihoods_UV.log2(), dim=(0, 2, 3)) / (latent_H_UV * latent_W_UV)

        self.channel_wise_entropy_Y += entropy_Y.squeeze()
        self.channel_wise_entropy_UV += entropy_UV.squeeze()
