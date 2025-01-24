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
import torch
import einops
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Dict
from .MCM_phases import MCM_phase0, MCM_phase1, MCM_phase2, MCM_phase3
from .utils import ContextUtils, HyperToContext9x1b
from src.codec.common import determinism_on_eval, Decisions


class Context(nn.Module, ContextUtils):
    def __init__(self, chs, quantize_func, skip_cube_thr=None, cube_size=None, cube_chan=None, num_decode_chs:int=None):
        super(Context, self).__init__()
        self.quantize_func = quantize_func
        self.chs = chs
        self.MCM = nn.ModuleList([])
        self.MCM.append(MCM_phase0(chs))
        self.MCM.append(MCM_phase1(chs))
        self.MCM.append(MCM_phase2(chs))
        self.MCM.append(MCM_phase3(chs))

        self.down_shuffle_conv_hyper = HyperToContext9x1b()
        self.skip_cube_thr = skip_cube_thr
        self.cube_size = cube_size
        self.cube_chan = cube_chan
        self.num_decode_chs = num_decode_chs
        
    def export_models(self, output_dir: str, opset_version=11):
        os.makedirs(output_dir, exist_ok=True)
        device = next(self.parameters()).device
                  
        prev_reco = None #if si ==0 else torch.rand([1, self.chs//2, 256, 256], device=device)
        for i, m in enumerate(self.MCM):
            device = next(m.parameters()).device
            d1 = torch.rand([1, self.chs, 256, 256], device=device)
            args = (d1)
            input_names=[f'p{i}']
            dynamic_axes={f'p{i}': [2,3]}
            if prev_reco is not None:
                y_name = f'{i-1}' if i == 1 else f'{i-1}-0'
                input_names.append(f'y[{y_name}]')
                dynamic_axes[f'y[{y_name}]'] = [2,3]
                args = (d1, prev_reco)
            torch.onnx.export(m, 
                        args, 
                        os.path.join(output_dir, f"stage{i}.onnx"),
                        export_params = True, opset_version = opset_version,
                        input_names=input_names,
                        output_names=[f'y[{i}]'],
                        dynamic_axes=dynamic_axes
                        )
            
            prev_reco = d1 if prev_reco is None else torch.cat( (prev_reco, d1), dim=1)
            
        

    @determinism_on_eval
    def pred(self, input:torch.Tensor, hyper_params:torch.Tensor, tool_params:Dict = None, h_ls= None, w_ls= None) -> \
        Union[torch.Tensor, Tuple[torch.Tensor,torch.Tensor,torch.Tensor]]:
        """ Single slice get dequantid y, pred_implicit, residual

        Args:
            input (torch.Tensor): Residual
            hyper_params (torch.Tensor): Pred_explicit information that getted from Hyper decoder.
            slice_rec(torch.Tensor): Dequantized y of the previous slice.
            slice_index(int): Slice index
            tool_params (dict, optional): Rvs and res_skip parameters, defaults to None..

        Returns:
            torch.Tensor: Dequantized y
            torch.Tensor: Implicit prediction of latent y
            torch.Tensor: Quantized, residual
        """

        hyper_param_list = self.down_shuffle_conv_hyper(hyper_params)
        input_list = ContextUtils.down_shuffle(input)

        mu_list = list()
        y_hat_list = list()
        resi_list = list()
        resi_dq_list = list()
        cubeflag_list = list()
        spatial_params = torch.zeros_like(input_list[0], device=input.device)
        cubeflagsfull_list = list()

        for stage_id_cur, stage_model in enumerate(self.MCM):
            mu_stage = stage_model(hyper_param_list[stage_id_cur], spatial_params)

            if tool_params is not None:
                tools_cur_stage = {
                    'mask2': tool_params["mask2"][stage_id_cur] if tool_params["mask2"] is not None else None,
                    'quantizer': tool_params["quantizer"][stage_id_cur] if tool_params["quantizer"] is not None else None
                }

                diff = input_list[stage_id_cur] - mu_stage
                if self.num_decode_chs is not None:
                    diff[:,self.num_decode_chs:] = 0
                resi_dq, resi_q = self.quantize_func(data=diff, tool_params=tools_cur_stage)
                #generate cube flag, using cube flag to control skip
                if self.cube_size is not None: # TODO: enable cube in training
                    cubeflag = self.gen_skip_cubeflag(resi_dq, diff, stage_id_cur, h_ls, w_ls)
                    cubeflagfull = self.convert_cubeflag_map(cubeflag, diff.shape)
                    cubeflagsfull_list.append(cubeflagfull)
                    tools_cur_stage['mask2'] = torch.logical_or(tools_cur_stage['mask2'], cubeflagfull)
                tools_cur_stage['mask2'] = self._mask_redundant_padding_mask(tools_cur_stage['mask2'], stage_id_cur, h_ls, w_ls)
                resi_dq, resi_q = self.quantize_func(data=diff, tool_params=tools_cur_stage)
                resi_list.append(resi_q)
                resi_dq_list.append(resi_dq)
                if self.cube_size is not None: # TODO: enable cube in training
                  cubeflag_list.append(cubeflag)
            else:
                resi_dq = input_list[stage_id_cur]
            y_hat_single_stage = resi_dq + mu_stage
            y_hat_list.append(y_hat_single_stage)
            spatial_params = torch.cat(y_hat_list, dim=1)
            mu_list.append(mu_stage)

        # Upshuffle the dequantized y, see figure E.4 in WD.
        y_hat = ContextUtils.up_shuffle(y_hat_list)
        if len(resi_list) > 0:
            mu = ContextUtils.up_shuffle(mu_list)
            resi = ContextUtils.up_shuffle(resi_list)
            resi_dq = ContextUtils.up_shuffle(resi_dq_list)
            cube_flag = None
            if self.cube_size is not None: # TODO: enable cube in training, ugly code
              part0, part3, part1, part2 = cubeflag_list
              cube_flag = torch.cat((part0, part1, part2, part3), dim=1)
            return y_hat, mu, resi, resi_dq, cube_flag
        else:
            return y_hat

    def decompress(self, resi:torch.tensor, params:torch.tensor):
        """ Decompress the dequantized y.

        Args:
            resi (torch.Tensor): Dequantized residuals
            params (torch.Tensor): Pred_explicit information that getted from Hyper decoder.

        Returns:
            torch.Tensor: Dequantized  y
        """
        h, w = resi.shape[-2:]

        resi_padding = F.pad(resi, (0, w % 2, 0, h % 2))
        #params_padding = F.pad(params, (0, w % 2, 0, h % 2))

        return self.pred(resi_padding, params, None)[:, :, 0:h, 0:w]

    def forward(self, y:torch.Tensor, hyper_params:torch.Tensor, tool_params:Dict = None):
        """ Prediction process of MCM in encoder.

        Args:
            y (torch.Tensor): Latent y
            hyper_params (torch.Tensor): Pred_explicit information that getted from Hyper decoder.
            tool_params (dict, optional): Rvs and res_skip parameters.

        Returns:
            torch.Tensor: Dequantized y
            torch.Tensor: Implicit prediction of latent y
            torch.Tensor: Quantized residual
        """

        h, w = y.shape[-2:]

        #padding
        y_padding = F.pad(y, (0, w % 2, 0, h % 2))

        if tool_params is None:
            mask2 = torch.ones_like(y, dtype=torch.bool)
            tool_params = {"scale1": None,
                           "mask2": mask2,
                           "adaptQuant" : False,
                           "quantizer": None}
        else:
            quantizer_params = tool_params.get('quantizer')
            quantizer_stage = None
            for mn,mv in quantizer_params.items():
                if mv is not None:
                    for k,v in mv.items():
                        if isinstance(v, torch.Tensor):
                            vp = F.pad(v,  (0, w % 2, 0, h % 2))
                            vp_stage = self.down_shuffle(vp)
                        else:
                            vp_stage = [vp]*self.STAGES_COUNT
                        if quantizer_stage is None:
                            quantizer_stage = [Decisions() for _ in range(len(vp_stage))]
                        for i, v in enumerate(vp_stage):
                            mnv = quantizer_stage[i].get(mn, Decisions())
                            mnv[k] = v
                            quantizer_stage[i].update({mn: mnv})
            tool_params['quantizer'] = quantizer_stage
                
        mask2 = tool_params.get('mask2')
        if mask2 is not None:
            mask2_padding = F.pad(mask2, (0, w % 2, 0, h % 2))
            mask2_ds = ContextUtils.down_shuffle(mask2_padding)        #Resi Skip
            tool_params['mask2'] = mask2_ds

        y_rec, mu, resi_q, resi_dq, cube_flag = self.pred(y_padding, hyper_params, tool_params, h, w)

        resi_q = resi_q[:, :, 0:h, 0:w]
        resi_dq = resi_dq[:, :, 0:h, 0:w]
        mu = mu[:, :, 0:h, 0:w]
        y_rec = y_rec[:, :, 0:h, 0:w]

        return y_rec, mu, resi_q, resi_dq, cube_flag

    def gen_skip_cubeflag(self, y_hat: torch.Tensor, y_org: torch.Tensor, stage_id_cur, h_ls, w_ls) -> torch.Tensor:
        diff_yhat = torch.abs(y_hat - y_org)
        diff_yhat = self._mask_redundant_padding_tensor(diff_yhat, stage_id_cur, h_ls, w_ls)
        N, C, H, W = diff_yhat.shape
        diff_yhat = diff_yhat.reshape(1, N, C, H, W)
        cube_size = self.cube_size
        cube_chan = self.cube_chan
        h_pad = ((H + cube_size - 1) // cube_size) * cube_size - H
        w_pad = ((W + cube_size - 1) // cube_size) * cube_size - W
        diff_yhat = F.pad(diff_yhat, (0, w_pad, 0, h_pad), value=0)
        maxpool = torch.nn.MaxPool3d((cube_chan, cube_size, cube_size), (cube_chan, cube_size, cube_size), 0)
        cubeflag = (maxpool(diff_yhat) > self.skip_cube_thr) # skip_mask = cubeflag ! skip_mask
        cubeflag = ~cubeflag[0,:,:,:,:]
        return cubeflag

    def convert_cubeflag_map(self, init_map: torch.Tensor, output_shape: torch.Size) -> torch.Tensor:
        _,y_chan, y_hei, y_wid = output_shape
        cube_flags_full = einops.repeat(init_map, 'a b c d -> a (b repeat1) (c repeat2) (d repeat3)', repeat1=y_chan, repeat2=self.cube_size, repeat3=self.cube_size)
        # TODO: check repeat is right?
        cube_flags_full = ~cube_flags_full[:, :y_chan, :y_hei, :y_wid]
        return cube_flags_full

    def _mask_redundant_padding_tensor(self, input_tensor, stage_id, h_ls, w_ls):
        if h_ls % 2 == 1 and stage_id in [1, 3]:
            input_tensor[:, :, -1, :] = 0
        if w_ls % 2 == 1 and stage_id in [1, 2]:
            input_tensor[:, :, :, -1] = 0
        return input_tensor

    def _mask_redundant_padding_mask(self, input_mask, stage_id, h_ls, w_ls):
        if h_ls % 2 == 1 and stage_id in [1, 3]:
            input_mask[:, :, -1, :] = False
        if w_ls % 2 == 1 and stage_id in [1, 2]:
            input_mask[:, :, :, -1] = False
        return input_mask
