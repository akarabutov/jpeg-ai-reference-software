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

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.codec.coding_tools.core_models.CCS_SGMM.sep_chan_tool import \
    SepChannelsSGMMTool

from ...common import Decisions, Image, tiling
from ..core_models.base.interface import CoreModelBase
from ..core_models.CCS_SGMM import CcsGvaeMultiTools, CcsGvaeSGMM
from ..interfaces import AttrsProxy, RdoPostProcInterface, ToolEngine
from .params import RdlrParamsBPP, RdlrParamsCommon
from .rdlr_comp_Y import RDLRCompY
from .rdlr_comp_UV import RDLRCompUV


class RDLR(RdoPostProcInterface):
    tile_min_size_for_MSSSIM = 176  # for MS-SSIM need tiles larger than 160. also keep alignment size ->  176 x 176

    def __init__(self, **kwargs):
        super(RDLR, self).__init__(**kwargs)
        self._params_rdlr = RdlrParamsCommon()
        self._params_bpp = RdlrParamsBPP()

        self.model_y = RDLRCompY(tile_min_size_for_MSSSIM=self.tile_min_size_for_MSSSIM)
        self.model_uv = RDLRCompUV(tile_min_size_for_MSSSIM=self.tile_min_size_for_MSSSIM)

        models_list = [self.model_y, self.model_uv]
        self._attrs_proxy_bpp = AttrsProxy(self._params_bpp.get_params_name_list(), models_list)

        self.supported_tools_types = [CcsGvaeMultiTools, CcsGvaeSGMM]

    def _params_loaded(self):
        # Set parameters of tiles for sub models.
        luma_tile_attr = self.generate_attrs_dict(self._params_rdlr.get_params_name_list(), 'Luma',
                                                  ['Luma'])
        self.model_y.set_attrs_value(luma_tile_attr)
        chroma_tile_attr = self.generate_attrs_dict(self._params_rdlr.get_params_name_list(),
                                                    'Chroma', ['Chroma'])
        self.model_uv.set_attrs_value(chroma_tile_attr)

    def __calc_lr(self, num_samples: int):
        if self.LearningRateYAutomaticPerResolution:
            if num_samples >= (3840 * 2160):
                return 0.02
            elif num_samples >= (832 * 480):
                return 0.03
            else:
                return 0.05
        else:
            return self.LearningRateY

    def _replace_not_sane_slopes_with_mean(self, slope: float, means: dict):
        if (slope is None) or not (0 <= slope <= 90):
            bpp = int(self.get_target_bpp() * 100)
            if bpp in means.keys():
                return means[bpp]
            else:
                return means['other']
        else:
            return slope

    def _get_slopes_parameters(self):

        if self.lossTypeBDcurveSlope_data_source == 'dataimg_all':

            no_rdlr_sim_summary = pd.read_csv(self.lossTypeBDcurveSlope_df, index_col=0)
            image_filepath = self.get_owner_param('image_filepath')
            # image_filepath = self.image_filepath
            image_name = Path(image_filepath).stem

            # extract rd curve slope for current sequence and bpp
            current_seq_noRDLR_rate_points_df = no_rdlr_sim_summary.loc[
                no_rdlr_sim_summary.seq_name.str.contains(image_name), :].reset_index(
                    drop=True).copy()

            # per sequence and bpp
            current_bpp_curve_slopes = current_seq_noRDLR_rate_points_df[
                current_seq_noRDLR_rate_points_df.target_bpp == int(self.get_target_bpp() *
                                                                    100)].reset_index(
                                                                        drop=True).copy()[[
                                                                            'metric', 'slope'
                                                                        ]]
            # transform to series for easy elem access
            current_bpp_curve_slopes = current_bpp_curve_slopes.set_index('metric')['slope']

            # convert angles to rad
            current_bpp_curve_slopes = current_bpp_curve_slopes / 180 * np.pi

        else:  # only other case is self.lossTypeBDcurveSlope_data_source == 'single_seq_bpp_slopes'

            # sanity check for slopes given by cmd line.
            # if not
            #    0 < slope < 90
            # replace it with mean of dataset value (from JVET CfP) for current bpp

            mean_msssimY_slopes = {
                6: 1.629650,
                12: 1.200354,
                25: 0.785844,
                50: 0.372070,
                75: 0.175778,
                'other': 0.832739,
            }
            mean_psnrY_slopes = {
                6: 74.868811,
                12: 76.474952,
                25: 76.252456,
                50: 69.157414,
                75: 57.958624,
                'other': 70.942451,
            }
            mean_psnrU_slopes = {
                6: 64.131075,
                12: 62.749463,
                25: 63.404102,
                50: 58.369855,
                75: 44.556739,
                'other': 58.642247,
            }
            mean_psnrV_slopes = {
                6: 64.006979,
                12: 62.568593,
                25: 62.318648,
                50: 56.189655,
                75: 42.682759,
                'other': 57.553327,
            }

            lossTypeBDcurveSlope_msssimY_slope = self._replace_not_sane_slopes_with_mean(
                self.lossTypeBDcurveSlope_msssimY_slope, mean_msssimY_slopes)
            lossTypeBDcurveSlope_psnrY_slope = self._replace_not_sane_slopes_with_mean(
                self.lossTypeBDcurveSlope_psnrY_slope, mean_psnrY_slopes)
            lossTypeBDcurveSlope_psnrU_slope = self._replace_not_sane_slopes_with_mean(
                self.lossTypeBDcurveSlope_psnrU_slope, mean_psnrU_slopes)
            lossTypeBDcurveSlope_psnrV_slope = self._replace_not_sane_slopes_with_mean(
                self.lossTypeBDcurveSlope_psnrV_slope, mean_psnrV_slopes)

            slopes = {
                'msssim_Torch': lossTypeBDcurveSlope_msssimY_slope,
                'psnrY': lossTypeBDcurveSlope_psnrY_slope,
                'psnrU': lossTypeBDcurveSlope_psnrU_slope,
                'psnrV': lossTypeBDcurveSlope_psnrV_slope,
            }
            current_bpp_curve_slopes = pd.Series(data=slopes, index=slopes.keys())
            current_bpp_curve_slopes = current_bpp_curve_slopes / 180 * np.pi

        return current_bpp_curve_slopes

    @staticmethod
    def get_luma_tile_rate(rdlr_comp: RDLRCompY, tool: SepChannelsSGMMTool, decisions: Decisions,
                           orig: torch.Tensor, latent_UV_tile_with_overlap,
                           image_tile_UV_with_overlap, optimized_part_of_imgUVtile_rel_to_overlap):
        latent_Y_tile_with_overlap = latent_UV_tile_with_overlap.upscale(2)
        image_tile_Y_with_overlap = image_tile_UV_with_overlap.upscale(2)
        optimized_part_of_imgYtile_rel_to_overlap = optimized_part_of_imgUVtile_rel_to_overlap.upscale(
            2)
        # y_hat = decisions.get('y_hat')
        y_hat = decisions.get('y')

        lat_Y_tile = tiling.get_data(y_hat, latent_Y_tile_with_overlap)
        im_Y_tile = tiling.get_data(orig, image_tile_Y_with_overlap)

        cur_decisions_Y = Decisions()
        cur_decisions_Y['y'] = lat_Y_tile
        _ = tool._compress_z(cur_decisions_Y,
                             h=image_tile_Y_with_overlap.size.height,
                             w=image_tile_Y_with_overlap.size.width)

        cur_decisions_Y['psi'] = tool.hyper_decoder(cur_decisions_Y['z_hat'].float(),
                                                    h=image_tile_Y_with_overlap.size.height,
                                                    w=image_tile_Y_with_overlap.size.width)
        cur_decisions_Y['scale_log'] = tool.hyper_scale_decoder(cur_decisions_Y['z_hat'],
                                     h=image_tile_Y_with_overlap.size.height,
                                     w=image_tile_Y_with_overlap.size.width)

        tile_rate_ref_Y_part, _, _ = rdlr_comp.get_current_rate_point(
            tool,
            cur_decisions_Y,
            lat_Y_tile,
            im_Y_tile,
            optimized_part_of_imgYtile_rel_to_overlap,
            only_rate=True)
        return tile_rate_ref_Y_part

    def _process(self, tool: ToolEngine, orig_img: Image, decision: Decisions) -> Decisions:

        num_samples = orig_img.shape[-1] * orig_img.shape[-2]

        if num_samples < self.numSamples:

            slopes = self._get_slopes_parameters()
            self.model_y.current_bpp_curve_slopes = slopes
            self.model_uv.current_bpp_curve_slopes = slopes

            lr = self.__calc_lr(num_samples)

            cur_img = orig_img
            cur_img.to_YUV_()
            cur_img.convert_range_(tool.get_internal_data_range())
            cur_img.to_420_()
            img_y = cur_img.get_component('a')
            img_uv = torch.cat((cur_img.get_component('b'), cur_img.get_component('c')), dim=1)

            models_list = ['model_y', 'model_uv']
            img_list = [img_y, img_uv]

            sup_info = None
            calc_additional_rate = None
            _, _, img_height_y, img_width_y = img_y.shape
            _, _, img_height_uv, img_width_uv = img_uv.shape

            # RDLR Y
            self.get_profilers().start('RDLR Y')
            sep_chan_tool_y: SepChannelsSGMMTool = getattr(tool, 'model_y')
            sep_chan_tool_y.decoder.float()

            decision_z = decision.get('model_y').clone()

            decision_y_and_z = self.model_y.process_y(sep_chan_tool_y,
                                                      decision_z,
                                                      img_y,
                                                      lr=lr,
                                                      lr_decay=self.LearningRateYDecay)

            sup_info = decision_y_and_z.get('y_hat')[:, :, ::2, ::2]
            decision.get('model_y').update(decision_y_and_z)

            # keep Y rate for rdlr UV
            calc_additional_rate = partial(self.get_luma_tile_rate,
                                           rdlr_comp=self.model_y,
                                           decisions=decision_y_and_z,
                                           orig=img_y,
                                           tool=sep_chan_tool_y)
            self.get_profilers().finish('RDLR Y')

            # RDLR UV
            self.get_profilers().start('RDLR UV')
            sep_chan_tool_uv: SepChannelsSGMMTool = getattr(tool, 'model_uv')
            sep_chan_tool_uv.decoder.float()

            decision_z = decision.get('model_uv').clone()

            decision_y_and_z = self.model_uv.process_y(sep_chan_tool_uv,
                                                       decision_z,
                                                       img_uv,
                                                       sup_info=sup_info,
                                                       lr=lr,
                                                       lr_decay=self.LearningRateYDecay,
                                                       calc_additional_rate=calc_additional_rate)
            decision.get('model_uv').update(decision_y_and_z)
            self.get_profilers().finish('RDLR UV')

        return decision
