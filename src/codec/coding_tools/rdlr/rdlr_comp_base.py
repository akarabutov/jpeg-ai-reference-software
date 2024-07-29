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
from functools import partial
from typing import List, Tuple

import numpy as np
import torch
from pytorch_msssim import ms_ssim

from src.codec.coding_tools.core_models.CCS_SGMM.sep_chan_tool import \
    SepChannelsSGMMTool
from src.codec.components.entropy_coding.prob_models.gm import GMProbModel

from ...common import Decisions, tiling
from ...entropy_coding import ECLibLH, ECModule
from ..interfaces import AttrsProxy, BaseEngine, ToolEngine
from ..tiling import TileManager, TilingParams
from .params import RdlrCompParams, RdlrParamsBPP
##
from .utils import Substract


class RDLRCompBase(BaseEngine):
    tile_min_size_for_MSSSIM = 176  # for MS-SSIM need tiles larger than 160. also keep alignment size ->  176 x 176

    def __init__(self, **kwargs):
        super(RDLRCompBase, self).__init__(**kwargs)
        self._param_rdlr_bpp = RdlrParamsBPP()
        self._param_rdlr_comp = RdlrCompParams()
        self._tiling_params = TilingParams()
        self.tile_manager = TileManager(16, 16)
        self.tile_min_size_for_MSSSIM = kwargs.get('tile_min_size_for_MSSSIM',
                                                   self.tile_min_size_for_MSSSIM)
        self.downsample_image = kwargs.get('downsample_image', False)

        self._attrs_proxy_tile_params = AttrsProxy(self._tiling_params.get_params_name_list(),
                                                   self.tile_manager)

        # switch enabling to plot rd path taken in RDLR, for debugging.
        # empty list: disabled
        # Y or UV in list: do it for those components
        # self.trace_path = ['Y', 'UV']
        self.trace_path = []

    def process_z(self, tool: SepChannelsSGMMTool, decision: Decisions, h: int,
                  w: int) -> Decisions:

        cur_decision = decision.clone()
        best_decision = decision.clone()
        nochange = 0
        numBits_best = math.inf
        numIte = self.numIteZ
        logger = self.logger

        z_Y = decision.get('z')
        device = z_Y.device
        bb_Y, cc_Y, hh_Y, ww_Y = z_Y.shape
        filter_alf2_Y = Substract(bb_Y, cc_Y, hh_Y, ww_Y)
        filter_alf2_Y.to(device)
        optimizer = torch.optim.Adagrad([{'params': filter_alf2_Y.parameters()}], lr=0.05)
        with torch.enable_grad():
            for i in range(numIte):
                ac = ECLibLH()
                ec = ECModule(ac)
                ac.encode_init()
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.9
                z_new_Y = filter_alf2_Y(z_Y)
                z_hat_Y = z_new_Y + torch.clamp(z_new_Y.detach().round() - z_new_Y.detach(), -1, 1)
                cur_decision['z_hat'] = z_hat_Y
                tool.encode_z(ec, cur_decision, h, w)
                del cur_decision['scale_log']
                tool.encode_y(ec, cur_decision, h, w)

                # tool.encode(ec, cur_decision, components={'Y'})
                numBits_dummy = ac.get_total_bits()
                loss = numBits_dummy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                nochange += 1
                logger.debug(f'Z: {numBits_dummy}')
                if numBits_dummy < numBits_best:
                    best_decision = cur_decision.clone()
                    optimizer.step()
                    if i < 3:
                        optimizer.step()
                    numBits_best = numBits_dummy
                    nochange = 0
                if nochange == 7:
                    break
        best_decision['z_hat'] = best_decision['z_hat'].round()
        return best_decision

    def get_ref_rate_data(self,
                          tool: ToolEngine,
                          decisions: Decisions,
                          lat_tile,
                          im_tile,
                          optimized_part_of_imgtile_rel_to_overlap,
                          sup_info=None) -> List:
        pass

    def _index_to_scale(self, index, log_k, log_b):
        scale = torch.exp(index * log_k + log_b)
        return scale

    def process_y(self, tool: SepChannelsSGMMTool, decisions: Decisions, orig_tensor: torch.Tensor,
                  **kwargs) -> Tuple[Decisions, List]:
        if self.numIteY == 0:
            return decisions

        best_decisions = decisions.clone()

        lr = kwargs.get('lr')
        lr_decay = kwargs.get('lr_decay')

        sup_info = kwargs.get('sup_info', None)
        calc_additional_rate = kwargs.get('calc_additional_rate', None)

        PartialOptimizer = partial(torch.optim.Adagrad, lr=lr)
        PartialScheduler = partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=lr_decay)

        y = decisions.get('y')

        self.tile_manager.set_alignment_size(tool.get_alignment_size() *
                                             (2 if self.downsample_image else 1))

        tile_min_size = self.tile_min_size_for_MSSSIM * (
            2 if self.downsample_image else 1
        ) + self.tile_manager.numSamplesTileOverlap if self.lossTypeBDcurveSlope_ratioPSNR_MSSSIM else None
        self.tile_manager.setup_tiles_enc(orig_tensor.shape,
                                          self.tile_manager.calc_latent_shape(
                                              orig_tensor.shape, 1),
                                          minimum_tile_size=tile_min_size)

        image_tiles = self.tile_manager.image_tiles
        latent_tiles = self.tile_manager.latent_tiles

        if self.trace_path:
            self.path_Y = {}
            self.path_Y['bpp'] = []
            self.path_Y['psnrY'] = []
            self.path_Y['msssim'] = []
            self.path_Y['lr'] = []
            self.path_UV = {}
            self.path_UV['bpp'] = []
            self.path_UV['psnrU'] = []
            self.path_UV['psnrV'] = []
            self.path_UV['lr'] = []

        component = 'UV' if sup_info is not None else 'Y'
        self.logger.debug(f'RDLR {component} using these tiles:\n {repr(image_tiles)}')
        for image_tile_with_overlap, latent_tile_with_overlap in zip(image_tiles, latent_tiles):
            loss_best_tile = math.inf  # first rdlr iteration will redo qunatization of y to y_hat and hence effectively set loss to loss without RDLR. -> RDLR needs at least 2 iterations do change anything
            cur_decisions = decisions.clone()

            _, optimized_part_of_imgtile_rel_to_overlap = self.tile_manager.get_core_of_overlapping_image_tile(
                image_tile_with_overlap)
            optimized_part_of_lattile, optimized_part_of_lattile_rel_to_overlap = self.tile_manager.get_core_of_overlapping_latent_tile(
                latent_tile_with_overlap, image_tile_with_overlap)

            lat_suppinfo = tiling.get_data(
                sup_info, latent_tile_with_overlap) if sup_info is not None else None

            lat_tile = tiling.get_data(y, latent_tile_with_overlap)
            im_tile = tiling.get_data(orig_tensor, image_tile_with_overlap)

            best_y_tile = lat_tile.clone()

            cur_decisions['y'] = lat_tile

            # get starting rate point
            ref_data = self.get_ref_rate_data(tool, cur_decisions, lat_tile, im_tile,
                                              optimized_part_of_imgtile_rel_to_overlap,
                                              lat_suppinfo)
            if calc_additional_rate is not None:
                additional_rate = calc_additional_rate(
                    latent_UV_tile_with_overlap=latent_tile_with_overlap,
                    image_tile_UV_with_overlap=image_tile_with_overlap,
                    optimized_part_of_imgUVtile_rel_to_overlap=
                    optimized_part_of_imgtile_rel_to_overlap)
                ref_data[0] += additional_rate
                ref_data[3] += additional_rate

            data = tiling.get_data(lat_tile, optimized_part_of_lattile_rel_to_overlap)
            lat_tile_optimized_part = data.clone().requires_grad_()

            optimizer = PartialOptimizer([{'params': lat_tile_optimized_part}])
            scheduler = PartialScheduler(optimizer)

            if self.trace_path:
                self.path_Y['bpp'] += [float(ref_data[0])]
                self.path_Y['psnrY'] += [float(ref_data[1])]
                self.path_Y['msssim'] += [float(ref_data[2])]
                self.path_Y['lr'] += [optimizer.param_groups[0]['lr']]

            for i in range(self.numIteY):
                with torch.enable_grad():
                    lat_tile = lat_tile.detach()
                    lat_tile = tiling.assign_data(lat_tile, optimized_part_of_lattile_rel_to_overlap,
                                       lat_tile_optimized_part)

                    loss = self.rdlr_y_iteration(tool,
                                                 lat_tile,
                                                 im_tile,
                                                 cur_decisions,
                                                 optimizer,
                                                 scheduler,
                                                 optimized_part_of_imgtile_rel_to_overlap,
                                                 ref_data,
                                                 sup_info=lat_suppinfo)

                if loss < loss_best_tile:
                    best_y_tile = tiling.get_data(
                        cur_decisions.get('y').detach(), optimized_part_of_lattile_rel_to_overlap)
                    loss_best_tile = loss

            best_decisions['y'] = tiling.assign_data(best_decisions['y'], optimized_part_of_lattile, best_y_tile)

        _, _, h, w = orig_tensor.shape
        _ = tool.common_modules._compress_y(best_decisions['y'], best_decisions, h=h, w=w)

        if self.trace_path:
            path_bpp = self.path_Y['bpp']
            path_psnrY = self.path_Y['psnrY']
            path_msssimY = self.path_Y['msssim']
            path_lr = self.path_Y['lr']

            from pathlib import Path

            import matplotlib.pyplot as plt
            rdlr_path_dir = Path('rdlr_paths')
            rdlr_path_dir.mkdir(exist_ok=True)
            # video_name = self.get_parent().get_parent().video_filename
            # plt_image_name = f"{Path(video_name).stem}_Y_QP{self.get_qp()}.png"
            plt_image_name = 'rdlr_path.png'
            plt_image_path = rdlr_path_dir / plt_image_name

            fig, (ax_psnr, ax_msssim, ax_lr) = plt.subplots(1, 3, figsize=(60, 20))
            # plt.figure(figsize=(40,20))
            ax_psnr.plot(path_bpp, path_psnrY)
            ax_psnr.scatter(path_bpp, path_psnrY)
            for i in range(len(path_bpp)):
                ax_psnr.annotate(f'{i}', (path_bpp[i], path_psnrY[i]))
            ax_psnr.set_title('PSNR Y')
            # ax_psnr.set_xlim(0.055, 0.058 )
            # ax_psnr.set_xlim(0.05, 0.06 )
            # ax_psnr.set_ylim(23, 32 )

            ax_msssim.plot(path_bpp, path_msssimY)
            ax_msssim.scatter(path_bpp, path_msssimY)
            for i in range(len(path_bpp)):
                ax_msssim.annotate(f'{i}', (path_bpp[i], path_msssimY[i]))
            ax_msssim.set_title('MS-SSIM Y')
            # ax_msssim.set_xlim(0.055, 0.058 )
            # ax_msssim.set_xlim(0.05, 0.06 )
            # ax_msssim.set_ylim(0.93, 1.0 )

            ax_lr.plot(path_lr)
            ax_lr.set_yscale('log')
            ax_lr.set_title('lr')

            plt.savefig(plt_image_path)

        return best_decisions

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

        raise NotImplementedError

    def get_current_rate_point(self,
                               tool: SepChannelsSGMMTool,
                               y_component_with_overlap: torch.Tensor,
                               original: torch.Tensor,
                               decisions: Decisions,
                               optimized_part_of_imgtile_rel_to_overlap: tiling.Area,
                               sup_info: torch.Tensor = None):
        raise NotImplementedError

    def calc_distortion_loss(self,
                             original: torch.Tensor,
                             reconstruction: torch.Tensor,
                             metrics=['PSNR', 'MS-SSIM']):

        bits = 8
        max_val = (1 << bits) - 1

        results = {}

        rec_in = reconstruction.clone()  # do not modify original rec tensor

        a = original
        b = rec_in

        if 'PSNR' in metrics:
            mse = torch.mean((a - b)**2)
            mse += 1e-10  # avoid log(0) when identical
            psnr = 20 * np.log10(max_val) - 10 * mse.log10()
            results['PSNR'] = psnr

        if 'MS-SSIM' in metrics:
            msssim = ms_ssim(a, b, data_range=max_val)
            results['MS-SSIM'] = msssim

        return results
