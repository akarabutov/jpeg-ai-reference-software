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

from typing import List, Tuple

import numpy as np
import torch

from src.codec.coding_tools.core_models.CCS_SGMM.sep_chan_tool import \
    SepChannelsSGMMTool

from src.codec.common import Decisions, tiling
##
from src.codec.entropy_coding import ECLibLH, ECModule
from .rdlr_comp_base import RDLRCompBase


class RDLRCompUV(RDLRCompBase):
    def __init__(self, **kwargs):
        super(RDLRCompUV, self).__init__(downsample_image=True, **kwargs)

    def rdlr_y_iteration(self,
                         tool: SepChannelsSGMMTool,
                         y_component_with_overlap: torch.Tensor,
                         original: torch.Tensor,
                         cur_decisions: Decisions,
                         optimizer,
                         scheduler,
                         optimized_part_of_imgtile_rel_to_overlap: tiling.Area,
                         ref_data: Tuple,
                         sup_info: torch.Tensor = None):

        logger = self.logger

        device = y_component_with_overlap.device

        cur_decisions['y'] = y_component_with_overlap
        tile_joint_rate_ref, tile_psnrU_ref, tile_psnrV_ref, tile_rate_ref_Y_part = ref_data

        expected_rate_UV, current_psnrU, current_psnrV = self.get_current_rate_point(
            tool,
            cur_decisions,
            y_component_with_overlap,
            original,
            optimized_part_of_imgtile_rel_to_overlap,
            sup_info=sup_info)

        expected_joined_rate = expected_rate_UV + tile_rate_ref_Y_part

        psnr_x_comp = -np.sin(self.current_bpp_curve_slopes.psnrU)
        psnr_y_comp = np.cos(self.current_bpp_curve_slopes.psnrU)
        normal_vec_psnrU = torch.tensor([psnr_x_comp, psnr_y_comp], device=device)
        psnr_x_comp = -np.sin(self.current_bpp_curve_slopes.psnrV)
        psnr_y_comp = np.cos(self.current_bpp_curve_slopes.psnrV)
        normal_vec_psnrV = torch.tensor([psnr_x_comp, psnr_y_comp], device=device)

        rate_log_diff = (expected_joined_rate.log() - tile_joint_rate_ref.log()).view(-1)
        psnrU_diff = (current_psnrU - tile_psnrU_ref).view(-1)
        psnrV_diff = (current_psnrV - tile_psnrV_ref).view(-1)

        change_vec_psnrU = torch.cat([rate_log_diff, psnrU_diff])
        change_vec_psnrV = torch.cat([rate_log_diff, psnrV_diff])

        loss_psnrU = -1.0 * torch.dot(normal_vec_psnrU.float(), change_vec_psnrU.float())
        loss_psnrV = -1.0 * torch.dot(normal_vec_psnrV.float(), change_vec_psnrV.float())

        loss = (loss_psnrU + loss_psnrV) / 2

        logger.debug(f'Y: {loss.item()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        return float(loss)

    def get_current_rate_point(self,
                               tool: SepChannelsSGMMTool,
                               decisions: Decisions,
                               y_component_with_overlap: torch.Tensor,
                               original: torch.Tensor,
                               optimized_part_of_imgtile_rel_to_overlap: tiling.Area,
                               sup_info: torch.Tensor = None):
        ac = ECLibLH()
        ec = ECModule(ac)
        ac.encode_init()

        orig_U, orig_V = torch.chunk(original, 2, dim=1)
        h, w = orig_U.shape[-2:]

        y = y_component_with_overlap.clone()
        params = decisions['psi'].clone()
        scale_log = decisions['scale_log'].clone()
        if decisions['scale_lin'] is not None:
            scale_lin = decisions['scale_lin'].clone()
        else:
            scale_lin = None
        _, _, hh, ww = y.shape
        resi_hat, resi_dq, y_hat = tool.common_modules._compress_ar_scale(y.clone(), params, scale_log, True, scale_lin)
        means_hat = y_hat - tool.common_modules.quantizer.dequantize(resi_hat)

        y_new = y_hat
        if sup_info is not None:
            y_new2 = torch.cat((sup_info.detach(), y_new), dim=1)
        else:
            y_new2 = y_new

        rec_UV_in = tool.decoder(y_new2, h=h, w=w)

        resi = y - means_hat
        y_likelihoods = tool.common_modules.entropy._likelihood(resi, scale_log,
                                                 torch.zeros_like(scale_log))
        y_likelihoods = tool.common_modules.entropy.likelihood_lower_bound(y_likelihoods)

        measurements = {'x_hat': rec_UV_in, 'likelihoods': {'y': y_likelihoods, 'z': []}}
        rate_uv = torch.sum(-torch.log2(measurements['likelihoods']['y']))

        rec_U_in = rec_UV_in[:, 0:1, :, :]
        rec_V_in = rec_UV_in[:, 1:2, :, :]
        rec_U_in_optimized_part = tiling.get_data(rec_U_in,
                                                  optimized_part_of_imgtile_rel_to_overlap)
        rec_V_in_optimized_part = tiling.get_data(rec_V_in,
                                                  optimized_part_of_imgtile_rel_to_overlap)
        orig_U_optimized_part = tiling.get_data(orig_U, optimized_part_of_imgtile_rel_to_overlap)
        orig_V_optimized_part = tiling.get_data(orig_V, optimized_part_of_imgtile_rel_to_overlap)
        metrics_U = self.calc_distortion_loss(orig_U_optimized_part, rec_U_in_optimized_part,
                                              ['PSNR'])
        metrics_V = self.calc_distortion_loss(orig_V_optimized_part, rec_V_in_optimized_part,
                                              ['PSNR'])
        current_psnrU = metrics_U['PSNR']
        current_psnrV = metrics_V['PSNR']

        tool.encode_z(ec, decisions, h, w)

        test_numBits_UV = ac.get_total_bits() + rate_uv
        img_shape = self.get_original_img_shape()
        num_pixels = img_shape[-1] * img_shape[-2]
        test_bpp_UV = test_numBits_UV / num_pixels

        return test_bpp_UV, current_psnrU, current_psnrV

    def get_ref_rate_data(self,
                          tool: SepChannelsSGMMTool,
                          decisions: Decisions,
                          lat_tile,
                          im_tile,
                          optimized_part_of_imgtile_rel_to_overlap,
                          sup_info=None) -> List:
        tile_rate_ref_U_part, tile_psnrU_ref, tile_psnrV_ref = self.get_current_rate_point(
            tool,
            decisions,
            lat_tile,
            im_tile,
            optimized_part_of_imgtile_rel_to_overlap,
            sup_info=sup_info)
        tile_joint_rate_ref = tile_rate_ref_U_part
        ref_data = [tile_joint_rate_ref, tile_psnrU_ref, tile_psnrV_ref, 0]
        return ref_data
