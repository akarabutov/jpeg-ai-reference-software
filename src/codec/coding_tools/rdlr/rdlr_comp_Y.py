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
from src.codec.bitstream_structure import BitstreamStructure
from .rdlr_comp_base import RDLRCompBase


class RDLRCompY(RDLRCompBase):
    def __init__(self, **kwargs):
        super(RDLRCompY, self).__init__(downsample_image=False, **kwargs)

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
        tile_rate_ref_Y_part, tile_psnrY_ref, tile_msssimY_Torch_ref = ref_data

        new_expected_rate_Y, current_psnrY, current_msssimY = self.get_current_rate_point(
            tool, cur_decisions, y_component_with_overlap, original,
            optimized_part_of_imgtile_rel_to_overlap)

        if self.trace_path:
            self.path_Y['bpp'] += [float(new_expected_rate_Y)]
            self.path_Y['psnrY'] += [float(current_psnrY)]
            self.path_Y['msssim'] += [float(current_msssimY)]
            self.path_Y['lr'] += [optimizer.param_groups[0]['lr']]

        psnr_x_comp = -np.sin(self.current_bpp_curve_slopes.psnrY)
        psnr_y_comp = np.cos(self.current_bpp_curve_slopes.psnrY)
        normal_vec_psnr = torch.tensor([psnr_x_comp, psnr_y_comp], device=device)
        msssim_x_comp = -np.sin(self.current_bpp_curve_slopes.msssim_Torch)
        msssim_y_comp = np.cos(self.current_bpp_curve_slopes.msssim_Torch)
        normal_vec_msssim = torch.tensor([msssim_x_comp, msssim_y_comp], device=device)

        rate_log_diff = (new_expected_rate_Y.log() - tile_rate_ref_Y_part.log()).view(-1)
        psnrY_diff = (current_psnrY - tile_psnrY_ref).view(-1)
        msssimY_diff = (current_msssimY - tile_msssimY_Torch_ref).view(-1)

        change_vec_psnr = torch.cat([rate_log_diff, psnrY_diff])
        change_vec_msssim = torch.cat([rate_log_diff, msssimY_diff])

        loss_psnr = -1.0 * torch.dot(normal_vec_psnr.float(), change_vec_psnr.float())
        loss_msssim = -20.0 * torch.dot(normal_vec_msssim.float(), change_vec_msssim.float())

        loss = loss_msssim * self.lossTypeBDcurveSlope_ratioPSNR_MSSSIM + (
            1 - self.lossTypeBDcurveSlope_ratioPSNR_MSSSIM) * loss_psnr

        logger.debug(f'Y: {loss.item()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        return loss.item()

    def get_current_rate_point(self,
                               tool: SepChannelsSGMMTool,
                               decisions: Decisions,
                               y_component_with_overlap: torch.Tensor,
                               original: torch.Tensor,
                               optimized_part_of_imgtile_rel_to_overlap: tiling.Area,
                               sup_info: torch.Tensor = None,
                               only_rate: bool = False):


        bs = BitstreamStructure(ECLibLH)
        bs.encode_init()
        ec = ECModule(bs)

        _, _, h, w = original.shape

        y = y_component_with_overlap.clone()
        scale_log = decisions['scale_log'].clone()
        resi_hat, residual_dq, y_hat = tool.common_modules._compress_ar_scale(y.clone(), decisions)
        means_hat = y_hat - resi_hat

        y_new = y_hat
        if sup_info is not None:
            y_new2 = torch.cat((sup_info[:, :, ::2, ::2].detach(), y_new), dim=1)
        else:
            y_new2 = y_new

        rec_Y_in = tool.decoder(y_new2, h=h, w=w)

        resi = y - means_hat
        scale_log2 = scale_log / pow(2,  tool.common_modules.hyper_scale_decoder.sigma_out_precision) # undo scaling to precision of HSD
        scale_lin = self._index_to_scale(scale_log2, tool.common_modules.log_k, np.log(tool.common_modules.sigma_quant_min))

        y_likelihoods = tool.common_modules.entropy(resi, scale_lin, torch.zeros_like(scale_lin))

        measurements = {'x_hat': rec_Y_in, 'likelihoods': {'y': y_likelihoods, 'z': []}}
        rate_y = torch.sum(-torch.log2(measurements['likelihoods']['y']))

        if not only_rate:
            rec_Y_in_optimized_part = tiling.get_data(rec_Y_in,
                                                      optimized_part_of_imgtile_rel_to_overlap)
            original_optimized_part = tiling.get_data(original,
                                                      optimized_part_of_imgtile_rel_to_overlap)
            metrics_Y = self.calc_distortion_loss(original_optimized_part, rec_Y_in_optimized_part,
                                                  ['PSNR', 'MS-SSIM'])
            current_psnrY = metrics_Y['PSNR']
            current_msssimY = metrics_Y['MS-SSIM']

        tool.common_modules.encode_z(ec, decisions, h, w, in_rdlr=True)

        test_numBits_Y = bs.encode_term() + rate_y
        img_shape = self.get_original_img_shape()
        num_pixels = img_shape[-1] * img_shape[-2]
        test_bpp_Y = test_numBits_Y / num_pixels

        if only_rate:
            return test_bpp_Y, 0, 0
        else:
            return test_bpp_Y, current_psnrY, current_msssimY

    def get_ref_rate_data(self,
                          tool: SepChannelsSGMMTool,
                          decisions: Decisions,
                          lat_tile,
                          im_tile,
                          optimized_part_of_imgtile_rel_to_overlap,
                          sup_info=None) -> List:
        tile_rate_ref_Y_part, tile_psnrY_ref, tile_msssimY_Torch_ref = self.get_current_rate_point(
            tool, decisions, lat_tile, im_tile, optimized_part_of_imgtile_rel_to_overlap)
        ref_data = [tile_rate_ref_Y_part, tile_psnrY_ref, tile_msssimY_Torch_ref]
        return ref_data
