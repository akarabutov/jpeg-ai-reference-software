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
from src.codec.entropy_coding import create_lh_ecmodule
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


        _, _, h, w = original.shape
        common_modules = tool.common_modules

        # Same residual derivation as the encoder: MCM prediction, gain unit, RVS and skip mask.
        # resi_q holds the symbols that go to me-tANS, with a straight-through round so that the
        # gradient reaches y.
        resi_q, _, y_hat, _ = common_modules._compress_ar_scale(y_component_with_overlap,
                                                                self._ar_scale_decisions(decisions))

        # Residual rate under the coder's own model: scale_log is the integer log-domain sigma
        # index (after gain and RVS), which the entropy model maps to sigma itself in table mode.
        scale_idx = decisions['scale_log']
        if scale_idx.is_floating_point():
            scale_idx = scale_idx.round().to(torch.int32)
        y_likelihoods = common_modules.entropy(resi_q, scale_idx.to(resi_q.device), None)
        rate_y = torch.sum(-torch.log2(y_likelihoods))

        # z is not optimized, so its rate is a constant measured once per image (prepare_rate_model).
        test_numBits_Y = rate_y + self.z_bits
        test_bpp_Y = test_numBits_Y / (h * w)

        if only_rate:
            return test_bpp_Y, 0, 0

        rec_Y_in = tool.decoder(y_hat, h=h, w=w)
        rec_Y_in_optimized_part = tiling.get_data(rec_Y_in, optimized_part_of_imgtile_rel_to_overlap)
        original_optimized_part = tiling.get_data(original, optimized_part_of_imgtile_rel_to_overlap)
        metrics_Y = self.calc_distortion_loss(original_optimized_part, rec_Y_in_optimized_part,
                                              ['PSNR', 'MS-SSIM'])
        return test_bpp_Y, metrics_Y['PSNR'], metrics_Y['MS-SSIM']

    @staticmethod
    def _ar_scale_decisions(decisions: Decisions) -> Decisions:
        """Inputs of _compress_ar_scale built the same way as in
        CommonEncDecModules.compress_ar_scale_tile, but for the whole image."""
        ans = Decisions()
        ans['psi'] = decisions['psi']
        ans['skip_scale_log'] = decisions['skip_scale_log']
        quantizer = decisions['quantizer']
        ans['quantizer'] = Decisions()
        ans['quantizer']['gain_unit'] = Decisions()
        if 'rvs' in quantizer:
            ans['quantizer']['rvs'] = {'scales': quantizer['rvs']['scales']}
        if 'qual_map' in quantizer:
            ans['quantizer']['qual_map'] = {'qp_map': quantizer['qual_map']['qp_map']}
        return ans

    def prepare_rate_model(self, tool: SepChannelsSGMMTool, decisions: Decisions,
                           orig_tensor: torch.Tensor) -> None:
        _, _, h, w = orig_tensor.shape
        ec = create_lh_ecmodule()
        tool.common_modules.encode_z(ec, decisions, h, w, in_rdlr=True)
        self.z_bits = float(ec.get_total_bits())

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
