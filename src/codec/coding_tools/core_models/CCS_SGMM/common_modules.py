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
import os
import torch
import numpy as np
from time import time
from typing import Tuple
from src.codec.common import (Decisions, determinism, disable_tf32, disable_torch_random, TensorOps, tiling)
from src.codec.coding_tools.interfaces import ToolEngine
from src.codec.coding_tools.skip_ls import SkipModeCoder
from src.codec.coding_tools.quantization import Quantizer
from src.codec.components import (GMProbModel, LaplacianProbModel,
                                  Context, FactorizedProbModel,
                                  HyperDecoderFactory, HyperEncoderFactory,
                                  HyperScaleDecoderFactory)
from src.codec.components import QuantModule
from src.codec.components.contexts.utils import Upsample_proc
from .params import CcsSharedModelsParams
from src.codec.entropy_coding import ECModule
from src.codec.common.tiling import Area
from src.codec.coding_tools.tiling import TileManager, TileManagerHyper



class CommonEncDecModules(ToolEngine):
    def __init__(self, chs_ls: int, skip_depth_step: bool, alignment_size: int, alignment_size_dec: int, *args, **kwargs):
        super(CommonEncDecModules, self).__init__(*args, signal_enabled_flag=False, **kwargs)

        self.tile_manager_hyper = None
        self.z_offset = 31
        self.z_range = 63

        self.chs_ls = chs_ls
        self.skip_depth_step = skip_depth_step
        self.alignment_size_dec = alignment_size_dec
        self.alignment_size = alignment_size
        self.quant_z = QuantModule()        
        self.skip_mode = SkipModeCoder(chs_ls, stream_header_part="pic_header")
        self.quantizer = Quantizer(chs=chs_ls, stream_header_part="pic_header")
        self.hyper_enc_factory = HyperEncoderFactory()
        self.hyper_dec_factory = HyperDecoderFactory()
        self.hyper_scale_dec_factory = HyperScaleDecoderFactory()
        self._params_shared = CcsSharedModelsParams(hyper_encoder=self.hyper_enc_factory.keys()[0],
                                                    hyper_encoder_choices=self.hyper_enc_factory.keys(),
                                                    hyper_decoder=self.hyper_dec_factory.keys()[0],
                                                    hyper_decoder_choices=self.hyper_dec_factory.keys(),
                                                    hyper_scale_decoder=self.hyper_scale_dec_factory.keys()[0],
                                                    hyper_scale_decoder_choices=self.hyper_scale_dec_factory.keys(),
                                                    **kwargs)
        self.tile_manager_hd = TileManager(alignment_size=self.alignment_size_dec*4,  # z_tile need alignment to 64, not 16
                                                    latent_downscale_factor_y=self.alignment_size_dec,
                                                    use_coding_headers=False)
        self.tile_manager_mcm = TileManager(alignment_size=self.alignment_size_dec*2,  # z_tile need alignment to 64, not 16
                                                    latent_downscale_factor_y=self.alignment_size_dec,
                                                    use_coding_headers=False)
        self.tile_manager_residual_bitstream = TileManager(alignment_size=self.alignment_size_dec*2,  # z_tile need alignment to 64, not 16
                                                    latent_downscale_factor_y=self.alignment_size_dec,
                                                    use_coding_headers=False)      
        self.tile_manager_hyper = TileManagerHyper(use_coding_headers=False)  
        self._register_load_state_dict_pre_hook(self._load_state_dict_hook)
        self.clipping_mode = 0
        
    def _params_loaded(self):
        if self.num_chs is None:
            self.num_chs = self.chs_ls
        self.num_chs = max(0, min(self.num_chs, self.chs_ls))
        if self.num_decode_chs is None:
            self.num_decode_chs = self.num_chs
        
    def build_model(self):
        self.log_k = (np.log(self.sigma_quant_max) - np.log(self.sigma_quant_min)) / (self.sigma_quant_level-1)
        
        """Initialization of each part of the codec
        """

        # self.decoder.to(torch.float32 if (self.device.type == 'cpu' or not self.training) else torch.float16)

        self.entropy = GMProbModel(scale_table=None,
                                    scale_level=self.sigma_quant_level,
                                    scale_max=self.sigma_quant_max,
                                    scale_min=self.sigma_quant_min,
                                    bound_offset=self.sigma_bound_offset
        )
        # Hyper part
        self.hyper_entropy = FactorizedProbModel(self.chs_ls, max_symbol=self.z_range - 1)
        self.hyper_encoder = self.hyper_enc_factory.create_instance(name=self.hyper_encoder_type,
                                                                    chs=self.chs_ls,
                                                                    skip_depth_step=self.skip_depth_step,
                                                                    is_abs=self.abs_in_hyperprior)
        self.hyper_decoder = self.hyper_dec_factory.create_instance(name=self.hyper_decoder_type,
                                                                    chs=self.chs_ls,
                                                                    skip_depth_step=self.skip_depth_step,
                                                                    num_out=1)
        self.hyper_scale_decoder = self.hyper_scale_dec_factory.create_instance(
            name=self.hyper_scale_decoder_type, 
            skip_depth_step=self.skip_depth_step,
            chs=self.chs_ls)

        if self.use_context_module:
            self.context = Context(
                    chs = self.chs_ls, 
                    quantize_func=self.quant_dequant,
                    skip_cube_thr = self.skip_mode.skip_cube_thr,
                    cube_size = self.skip_mode.cube_size,
                    cube_chan = self.skip_mode.cube_chan,
                    num_decode_chs=self.num_chs)        
        self._set_int_entropy_pipeline_params()

    def _set_int_entropy_pipeline_params(self):
        unscaled_sigma_precision = self.quantizer.unscaled_sigma_precision
        scaled_sigma_precision = self.quantizer.scaled_sigma_precision

        self.hyper_scale_decoder.sigma_idx_max_value = (self.sigma_quant_level - 1) * (2 ** unscaled_sigma_precision) - 1
        self.hyper_scale_decoder.sigma_out_precision = unscaled_sigma_precision

        self.skip_mode.scaled_sigma_precision = scaled_sigma_precision
        self.skip_mode.unscaled_sigma_precision = unscaled_sigma_precision
        self.entropy.set_param_precision(unscaled_sigma_precision)
        

    def quant_dequant(self, data:torch.Tensor, tool_params:dict = None) -> Tuple[torch.Tensor]:
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
        quantizer_pararms = tool_params["quantizer"]
        mask2_list = tool_params["mask2"]

        #AQ process
        data_s = data
        # Vrq and cwg quantize the residual
        data_q = self.quantizer.quantize_resi(data_s, quantizer_pararms)

        # skip block and mask padding boundary
        if mask2_list is not None:
            mask2_list = mask2_list.to(torch.bool)
            data_q[~mask2_list] = 0

        type_info = torch.iinfo(torch.int16)#int32
        data_q = torch.clamp(data_q, type_info.min, type_info.max)

        resi_rq = data_q - (data_q - data_q.round()).detach()
        
        data_dq = self.quantizer.dequantize_resi(resi_rq, quantizer_pararms)
    
        return data_dq, resi_rq        
        
    def update_entropy_model(self):
        if self.entropy._offset.numel() <= 0:
            # We preserved this code just to have time measurment of this process.
            # The same check will be performed in ec.decode_sdt()
            start = time()
            scale_table = self.entropy.get_scale_table()
            self.entropy.update_scale_table(scale_table)
            self.entropy.update()
            self.logger.debug(f'update scale table took: {time() - start}')
            self.logger.debug(
                'In the final model scale table should stored with model, no need to compute every time. Please contact Bytedance for assistance once a new model is trained.'
            )
            
    def _load_state_dict_hook(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        exclude_keys = ["epoch", "best_loss", "optimizer"]
        case_changes = dict()
        for k_full in state_dict.keys():
            if k_full.startswith(prefix):
                rest_k = k_full[len(prefix):]
                if 'vr_vec' in rest_k:
                    k_full_new = k_full.replace('vr_vec', 'quantizer.gain_unit')
                    case_changes[k_full] = k_full_new
                
        for o,n in case_changes.items():
            state_dict[n] = state_dict.pop(o)
        for k in exclude_keys:
            if k in state_dict:
                del state_dict[k]


    @determinism
    def compress_colocated_tiles(self,
                 img: torch.Tensor = None,
                 decisions: Decisions = {},
                 tile_info: tiling.ColocatedTiles = None) -> Decisions:

        # tile descriptions for assigning latents, do not include overlapping regions
        y_core_tile_desc, y_core_tile_desc_rel_to_overlap = self.owner.tile_manager_enc.get_core_of_overlapping_latent_tile(
            tile_info.y, tile_info.img, self.tile_manager_hd)
        z_core_tile_desc, z_core_tile_desc_rel_to_overlap = self.owner.tile_manager_enc.get_core_of_overlapping_latent_tile_z(
            tile_info.z, tile_info.img, self.tile_manager_hd)

        img_tile_data = tiling.get_data(img, tile_info.img)
        tile_height, tile_width = img_tile_data.shape[2:]
        tile_decision_y = Decisions()
        tile_decision_y['y'] = self.owner.encoder(img_tile_data, h=tile_height, w=tile_width, is_clip=self.clipping_mode)
        self._compress_z(tile_decision_y['y'], tile_decision_y, h=tile_height, w=tile_width)
        y_core_tile_data = tiling.get_data(tile_decision_y['y'], y_core_tile_desc_rel_to_overlap)
        tiling.assign_data(decisions['y'], y_core_tile_desc, y_core_tile_data)
        z_core_tile_data = tiling.get_data(tile_decision_y['z_hat'], z_core_tile_desc_rel_to_overlap)
        tiling.assign_data(decisions['z_hat'], z_core_tile_desc, z_core_tile_data)

    @determinism
    def hyper_decode_tile(self,
                 decisions: Decisions,
                 tile_info: tiling.ColocatedTiles,
                 shape_for_hyper_decoder: tiling.Area) -> None:

        decisions['tiles_hd'][tile_info.img] =  Decisions()
        decisions['tiles_hd'][tile_info.img]['z_hat'] = tiling.get_data( decisions['z_hat'], tile_info.z)

        decisions['tiles_hd'][tile_info.img]['psi'] = self.hyper_decoder(decisions['tiles_hd'][tile_info.img]['z_hat'], h=shape_for_hyper_decoder.size.height, w=shape_for_hyper_decoder.size.width)



    @determinism
    def encoder_get_scales(self,
                 decisions: Decisions,
                 img_shape: torch.Size) -> None:

        decisions['scale_log_origin'] = self.hyper_scale_decoder(decisions['z_hat'], img_shape[2], img_shape[3])
        decisions['scale_log'] = decisions.get('scale_log_origin')
        decisions['quantizer'] = self.quantizer.analyze(decisions, excl_list=['rvs'])
        decisions['skip_scale_log'] = self.quantizer.quantize_scale(decisions.get('scale_log'), decisions.get('quantizer'), excl_list=['rvs'])
        decisions['scale_log'] = decisions.get('skip_scale_log')
        decisions['quantizer'].update(self.quantizer.analyze(decisions, incl_list=['rvs']))
        decisions['scale_log'] = self.quantizer.quantize_scale(decisions.get('skip_scale_log'), decisions['quantizer'], incl_list=['rvs'])

    @determinism
    def compress_ar_scale_tile(self,
                 decisions: Decisions,
                 tile_info: tiling.ColocatedTiles = None,
                 cube_flag_list: list = None) -> None:

        if self.get_owner_param('region_residual_in_its_own_substream_flag'):
            y_core_tile_desc = tile_info.y
            y_core_tile_desc_rel_to_overlap = Area.from_x_y_width_height(0, 0, tile_info.y.size.width,
                                                                         tile_info.y.size.height)
        else:
            y_core_tile_desc, y_core_tile_desc_rel_to_overlap = self.tile_manager_mcm.get_core_of_overlapping_latent_tile(
                tile_info.y, tile_info.img, self.tile_manager_hd)
         
        psi_tile_data = decisions['tiles_mcm_resi'][tile_info.img]['psi']

        tile_decisions = Decisions()
        y_tile_data = tiling.get_data( decisions['y'], tile_info.y)
        tile_decisions['psi'] = psi_tile_data
        tile_decisions['skip_scale_log'] = tiling.get_data(decisions['skip_scale_log'], tile_info.y)

        # TODO: need to handle those for tools on?
        tile_decisions['quantizer'] = Decisions()
        tile_decisions['quantizer']['gain_unit'] = Decisions()
        if 'rvs' in decisions['quantizer']:
            tile_decisions['quantizer']['rvs'] = {}
            tile_decisions['quantizer']['rvs']['scales'] = tiling.get_data(decisions['quantizer']['rvs']['scales'], tile_info.y)        
        if 'qual_map' in decisions['quantizer']:
            tile_decisions['quantizer']['qual_map'] = {}
            tile_decisions['quantizer']['qual_map']['qp_map'] = tiling.get_data(decisions['quantizer']['qual_map']['qp_map'], tile_info.y)

        residual_q_tile_data, residual_dq_tile_data, y_hat_tile_data, cube_flag_tile_data  = self._compress_ar_scale(y_tile_data, tile_decisions)
        decisions['residual_tiles'][tile_info.y]= residual_q_tile_data 
        decisions['residual_dq_tiles'][tile_info.y]= residual_dq_tile_data
        decisions['y_hat_tiles'][tile_info.y]= y_hat_tile_data

        residual_dq_core_tile_data = tiling.get_data(residual_dq_tile_data, y_core_tile_desc_rel_to_overlap)
        tiling.assign_data(decisions['residual'], y_core_tile_desc, residual_dq_core_tile_data)

        residual_q_core_tile_data = tiling.get_data(residual_q_tile_data, y_core_tile_desc_rel_to_overlap)
        tiling.assign_data(decisions['residual_quant'], y_core_tile_desc, residual_q_core_tile_data)

        cube_flag_tile_area = tile_info.y.downscale(16)
        tiling.assign_data(decisions['cube_flag'], cube_flag_tile_area, cube_flag_tile_data)

        cube_flag_list.append(cube_flag_tile_data)


    @determinism
    def decompress_ar_scale_tile(self,
                 decisions: Decisions,
                 tile_info: tiling.ColocatedTiles) -> None:
        decisions['tiles_mcm_resi'][tile_info.img]['residual'] = decisions['residual_dq_tiles'][tile_info.y]
        self._decompress_ar_scale(decisions['tiles_mcm_resi'][tile_info.img])


    @determinism
    def encoder_skip_and_cubeflag_for_tiles(self, decisions: Decisions) -> None:

        if self.skip_mode.enabled:
            # Final data generation
            self.skip_mode.cube_flag = decisions['cube_flag']
            self.skip_mode.cube_flags_full = self.skip_mode.convert_cubeflag_map(self.skip_mode.cube_flag, decisions['y'].shape)

            # TODO: why is it needed? is this safe? (it will solve a mismatch, but does it affect performance?) -> are cubeflag/skip mask not merge correctly?
            # set residual parts that were skipped to 0 
            self.skip_mode._update_mask(decisions['skip_scale_log'])
            mask = self.skip_mode.mask()
            decisions['residual_quant'][~mask] = 0
            decisions['residual'][~mask] = 0


    @determinism
    def print_control_points_y_hat_psi(self, decisions: Decisions) -> None:

        control_points_psi = []
        control_points_y_hat = []
        for img_tile_desc in sorted(decisions['tiles_mcm_resi'].keys()):
            if 'y_hat' in decisions['tiles_mcm_resi'][img_tile_desc].keys():
                hash_y_hat = TensorOps.get_hash(decisions['tiles_mcm_resi'][img_tile_desc]['y_hat'])
                control_points_y_hat.append(f'{hash_y_hat}')
            if 'psi' in decisions['tiles_mcm_resi'][img_tile_desc].keys():
                hash_psi = TensorOps.get_hash(decisions['tiles_mcm_resi'][img_tile_desc]['psi'])
                control_points_psi.append(f'{hash_psi}')


        self.logger.info(f"The psi control point is {'-'.join(control_points_psi)}\n")
        self.logger.info(f"The y_hat control point is {'-'.join(control_points_y_hat)}\n")


    @determinism
    def merge_y_hat_overlaps_of_tiles(self, decisions: Decisions) -> None:

        y_hat = torch.zeros(self.owner.tile_manager_synthesis.latent_shape, device=self.device)
        for tile_descriptions in zip(self.owner.common_modules.tile_manager_mcm.image_tiles, self.owner.common_modules.tile_manager_mcm.latent_tiles):
            (
                img_tile_desc, 
                y_tile_desc, 
            ) = tile_descriptions
            if 'y_hat' in decisions['tiles_mcm_resi'][img_tile_desc].keys():
                # tile descriptions for assigning latents, do not include overlapping regions
                if self.owner.common_modules.tile_manager_mcm.region_residual_in_its_own_substream_flag:
                    y_core_tile_desc = y_tile_desc
                    y_core_tile_desc_rel_to_overlap = Area.from_x_y_width_height(0, 0, y_core_tile_desc.size.width,
                                                                                y_core_tile_desc.size.height)
                else:
                    y_core_tile_desc, y_core_tile_desc_rel_to_overlap = self.owner.common_modules.tile_manager_mcm.get_core_of_overlapping_latent_tile(
                        y_tile_desc, img_tile_desc, self.tile_manager_hd)

                y_core_tile_data = tiling.get_data(decisions['tiles_mcm_resi'][img_tile_desc]['y_hat'], y_core_tile_desc_rel_to_overlap)
                tiling.assign_data(y_hat, y_core_tile_desc, y_core_tile_data)

        return y_hat

    @determinism
    def extract_y_hat_for_synthesis_tiles(self, decisions: Decisions, y_hat: torch.Tensor) -> None:

        for tile_descriptions in zip(self.owner.tile_manager_synthesis.image_tiles, self.owner.tile_manager_synthesis.latent_tiles):
            (
                img_tile_desc, 
                y_tile_desc, 
            ) = tile_descriptions
            decisions['tiles_synthesis'][img_tile_desc] = Decisions()
            decisions['tiles_synthesis'][img_tile_desc]['y_hat'] = tiling.get_data(y_hat, y_tile_desc)

    @determinism
    def merge_psi_overlaps_of_tiles(self, decisions: Decisions) -> None:

        latent_shape_psi = self.tile_manager_hd.latent_shape_psi
        # latent_shape_psi = [*latent_shape_z[:2], latent_shape_z[2]*2, latent_shape_z[3]*2]
        psi = torch.zeros(latent_shape_psi, device=self.device)
        for tile_descriptions in zip(self.tile_manager_hd.image_tiles, self.tile_manager_hd.latent_tiles_psi):
            (
                img_tile_desc, 
                psi_tile_desc, 
            ) = tile_descriptions

            # tile descriptions for assigning latents, do not include overlapping regions
            # psi_tile_desc = z_tile_desc.upscale(2)
            if self.tile_manager_mcm.region_residual_in_its_own_substream_flag:
                psi_core_tile_desc = psi_tile_desc
                psi_core_tile_desc_rel_to_overlap = Area.from_x_y_width_height(0, 0, psi_tile_desc.size.width,
                                                                               psi_tile_desc.size.height)
            else:
                psi_core_tile_desc, psi_core_tile_desc_rel_to_overlap = self.tile_manager_hd.get_core_of_overlapping_latent_tile_psi(
                    psi_tile_desc, img_tile_desc, self.tile_manager_hd)
                # psi_core_tile_desc = z_core_tile_desc.upscale(2)
                # psi_core_tile_desc_rel_to_overlap = z_core_tile_desc_rel_to_overlap.upscale(2)

            psi_tile_data = decisions['tiles_hd'][img_tile_desc]['psi']
            # assert psi_tile_data.shape[2] == psi_tile_desc.size.height and psi_tile_data.shape[3] == psi_tile_desc.size.width
            psi_core_tile_data = tiling.get_data(decisions['tiles_hd'][img_tile_desc]['psi'], psi_core_tile_desc_rel_to_overlap)
            tiling.assign_data(psi, psi_core_tile_desc, psi_core_tile_data)

        return psi

    @determinism
    def extract_psi_for_mcm(self, decisions: Decisions, psi: torch.Tensor) -> None:

        for tile_descriptions in zip(self.tile_manager_mcm.image_tiles, self.tile_manager_mcm.latent_tiles_psi):
            (
                img_tile_desc, 
                psi_tile_desc, 
            ) = tile_descriptions
            # psi_tile_desc = z_tile_desc.upscale(2)
            decisions['tiles_mcm_resi'][img_tile_desc] = Decisions()
            decisions['tiles_mcm_resi'][img_tile_desc]['psi'] = tiling.get_data(psi, psi_tile_desc)


    # ============================================================
    # Compress part
    
    @determinism
    def _compress_ar_scale(self, y_hat, decisions: Decisions):
        y_org = y_hat.clone()
        height, width = y_hat.shape[-2:]
        psi = decisions.get('psi')
        if not self.use_context_module:
            psi = Upsample_proc(torch.chunk(psi, chunks=4, dim=1))[:, :, 0:height, 0:width]
        #scale_log = decisions['scale_log'] 
        skip_scale_log = decisions.get('skip_scale_log')
       
        quantizer_params = decisions.get('quantizer')
        # Final data generation
        mask2, _ = self.skip_mode.generate_skip_mask(skip_scale_log)

        tool_params = {"mask2": mask2, 'quantizer': quantizer_params}

        if self.use_context_module:
            y_rec, _, resi_q_full, resi_dq_full, cube_flag = self.context(y_org, psi, tool_params)
            # y_rec, _, resi_q_full, resi_dq_full, self.skip_mode.cube_flag = self.context(y_org, psi, tool_params)
            # self.skip_mode.cube_flags_full = self.skip_mode.convert_cubeflag_map(self.skip_mode.cube_flag, y_org.shape)
            # self.logger.info(f"The second control point is {TensorOps.get_hash(resi_dq_full)}\n")
        else:
            y_rec_resi, _ = self.quant_dequant(y_hat - psi, tool_params)
            y_rec_resi[:,self.num_chs:] = 0
            y_rec = y_rec_resi + psi
            # generate cube flag, using cube flag to get final skip mask
            cube_flag = self.skip_mode.gen_skip_cubeflag(y_rec, y_org)
            cube_flags_full = self.skip_mode.convert_cubeflag_map(cube_flag, y_org.shape)
            tool_params['mask2'] = torch.logical_or(mask2, cube_flags_full)
            resi_dq_full, resi_q_full = self.quant_dequant(y_hat-psi, tool_params)
            resi_dq_full[:,self.num_chs:] = 0
            resi_q_full[:,self.num_chs:] = 0
            y_rec = resi_dq_full + psi
            # self.logger.info(f"The second control point is {TensorOps.get_hash(resi_dq_full)}\n")

        return resi_q_full, resi_dq_full, y_rec, cube_flag

    @determinism
    def decompress_y_hat_to_image_tile(self,
                 decisions: Decisions,
                 tile_info: tiling.ColocatedTiles,
                 rec: torch.Tensor,
                 supp_info_uv: torch.Tensor = None) -> None:

        decisions2 = decisions['tiles_synthesis']
        tile_manager = self.owner.tile_manager_synthesis

        img_core_tile_desc, img_core_tile_desc_rel_to_overlap = tile_manager.get_core_of_overlapping_image_tile(tile_info.img)
        img_tile_data = self.owner.decompress(decisions2[tile_info.img], 
                                        h=tile_info.img.size.height, 
                                        w=tile_info.img.size.width, 
                                        sup_info=supp_info_uv)
        img_core_tile_data = tiling.get_data(img_tile_data, img_core_tile_desc_rel_to_overlap)
        tiling.assign_data(rec, img_core_tile_desc, img_core_tile_data)            



    @determinism
    def _compress_z(self, y: torch.Tensor, decision: Decisions, h: int, w: int) -> Decisions:
        """generate tensor in hyper latent space

        Args:
            decision (Decisions): a dict has result in latent space.
            h (int): height of the image.
            w (int): width of the image.

        Returns:
            decision (Decisions): a dict has result in hyper latent space.
        """

        with self.get_profilers_ctx('CCS compress_z'):
            y_pad = y.to(device=next(self.hyper_encoder.parameters()).device)
            z = self.hyper_encoder(y_pad, h=h, w=w, is_clip=self.clipping_mode)
            min_z_value = -self.z_offset
            max_z_value = self.z_range - self.z_offset - 1
            z = torch.clamp(z, min_z_value, max_z_value)
            z_hat, z_tilde = self.quant_z(z)

            type_info = torch.iinfo(torch.int8)
            assert(type_info.min <= min_z_value and max_z_value <= type_info.max)
            z_hat = z_hat.to(torch.int8)

        decision.update({'z_hat': z_hat, 'z_tilde': z_tilde, 'z': z})

        return decision
    
    def compress(self, y: torch.Tensor, h: int, w: int, decision = None, *args, **kwargs) -> Decisions:
        if decision is None:
            ans = Decisions()
        else:
            ans = decision 
        ans.update({'y': y})

        ans['cube_flag'] = torch.empty_like(self.skip_mode.gen_skip_cubeflag(ans['y'], ans['y']))   # TODO: derive shape directly
        ans['tiles_hd'] =  Decisions()
        ans['tiles_mcm_resi'] =  Decisions()
        ans['tiles_synthesis'] = Decisions()

        self.encoder_get_scales(ans, (1,1,h,w))

        self.get_profilers().start('CCS hyper_decoder')
        for tile_info in tiling.ColocatedTiles.iter_colocated_grids(self.tile_manager_hd):
            self.hyper_decode_tile(
                ans,
                tile_info,
                tile_info.img.downscale(self.get_owner_param('downsample_factor')))
        self.get_profilers().finish('CCS hyper_decoder')

        # would not needed when merging hd and MCM use same tiles
        # combine y_hat overlapping areas, replacing overlaps outside of core region of a tile with part of neighboring tile
        ans['psi'] = self.merge_psi_overlaps_of_tiles(ans)
        self.extract_psi_for_mcm(ans, ans['psi'])

        self.get_profilers().start('CCS compress_ar_scale')
        cube_flag_list = list()
        for tile_info in tiling.ColocatedTiles.iter_colocated_grids(self.tile_manager_mcm):
            self.compress_ar_scale_tile(
                ans,
                tile_info,
                cube_flag_list)
        self.skip_mode.cube_flag_list = cube_flag_list
        self.get_profilers().finish('CCS compress_ar_scale')

        self.encoder_skip_and_cubeflag_for_tiles(ans)        
        
        return ans
    
    # ============================================================
    # Decompress part
    
            
    def decompress(self, decisions: Decisions, h:int, w:int, *args, **kwargs) -> torch.Tensor:
        with self.get_profilers_ctx(f'{self.owner.name} hyper_decoder'):
            decisions['psi'] = self.hyper_decoder(decisions['z_hat'], h=h, w=w)
        self.logger.debug('Decoding y_hat')
        with torch.no_grad(), self.get_profilers_ctx(f'{self.owner.name}._decompress_ar_scale'):
            self._decompress_ar_scale(decisions)

        raise NotImplementedError # check if this is used
        return decisions['y_hat']

    @determinism
    def _decompress_ar_scale(self, decisions):

        rv_full = decisions['residual']
        params = decisions['psi']
        if not self.use_context_module:
            mean = Upsample_proc(torch.chunk(params,chunks=4,dim=1))[:,:,0:rv_full.shape[2],0:rv_full.shape[3]]
            y_hat = rv_full + mean
            decisions.update({'y_hat': y_hat})
        else:
            y_hat = self.context.decompress(rv_full, params)
            decisions.update({'y_hat': y_hat})
        # self.logger.info(f"The third control point is {TensorOps.get_hash(y_hat)}\n")            
        
        
    
    # ============================================================
    # Encode part
    
    
    @determinism
    def encode(self, ec: ECModule, decisions: Decisions, h: int = None, w: int = None, rev_order: bool = False) -> None:
        ht, wt = self.get_processed_img_shape()
        h = ht if h is None else h            
        w = wt if w is None else w
        if rev_order:
            self.encode_y(ec, decisions, h, w)
            self.quantizer.encode(ec, decisions.get('quantizer'), h=h, w=w)
            self.encode_z(ec, decisions, h, w)
        else:
            self.encode_z(ec, decisions, h, w)
            self.quantizer.encode(ec, decisions.get('quantizer'), h=h, w=w)        
            self.encode_y(ec, decisions, h, w)


    @determinism
    def encode_z(self, ec: ECModule, decision: Decisions, h: int, w: int, in_rdlr: bool = False):
        """generate bit stream in hyper latent space

        Args:
            ec (ECModule): entropy model.
            decision (Decisions): results in hyper latent space.
            h (int): height of the image.
            w (int): width of the image.
        """

        if 'z_hat' not in decision:
            y = decision['y']
            y = y.to(device=next(self.hyper_encoder.parameters()).device)
            z = self.hyper_encoder(y, h=h, w=w, is_clip=self.clipping_mode)
            z = torch.clamp(z, -self.z_offset, self.z_range - self.z_offset - 1)
            z_hat = z.round()
            decision['z'] = z
            decision['z_hat'] = z_hat
        else:
            z_hat = decision['z_hat']

        self.logger.info(f"The z_hat control point is {TensorOps.get_hash(z_hat)}\n")
        self._ac_encode_z(ec, z_hat, in_rdlr=in_rdlr)
        
        h_ls, w_ls = self.get_ls_shape(1)
        z_s = z_hat.shape[-2:]
        assert(h_ls==z_s[0] and w_ls==z_s[1]) 
            

    @determinism
    def encode_y(self, ec: ECModule, decision: Decisions, h: int, w: int):
        """generate bit stream in latent space

        Args:
            ec (ECModule): entropy mdoel.
            decision (Decisions): result in latent space.
            h (int): height of the image.
            w (int): width of the image.
        """

        residual = decision.get('residual_quant')
        scale_log = decision.get('scale_log')
        skip_scale_log = decision.get('skip_scale_log')
        # self.skip_mode.cube_flags_full = decision.get('cube_flags')
        self.logger.debug('Encoding quantized residual')
        
        self.skip_mode._update_mask(skip_scale_log)

        if self.tile_manager_residual_bitstream.enabled:
            self.logger.debug(
                f'Encode residual using these tiles: {repr(self.tile_manager_residual_bitstream.image_tiles_entropy_coding)}')
            tile_control_points2 = []
            tile_control_points3 = []
            tile_control_points5 = []
            tile_list = list(enumerate(zip(self.tile_manager_residual_bitstream.image_tiles_entropy_coding,
                                            self.tile_manager_residual_bitstream.latent_tiles_entropy_coding)))
            for tile_idx, (image_tile_area, latent_tile_area) in reversed(tile_list):

            # for image_tile_area, latent_tile_area in zip(reversed(self.tile_manager_residual_bitstream.image_tiles_entropy_coding),
            #                                 reversed(self.tile_manager_residual_bitstream.latent_tiles_entropy_coding)):

                # tile_to_be_encoded, _ = self.tile_manager_residual_bitstream.get_core_of_overlapping_latent_tile(
                #     latent_tile_area, image_tile_area)

                tile_to_be_encoded = latent_tile_area

                mask_tile, output_shape = self.skip_mode.mask_with_tile(tile_to_be_encoded)

                scale_log_tile =  tiling.get_data(scale_log, tile_to_be_encoded)
                residual_tile =  tiling.get_data(residual, tile_to_be_encoded)

                # scale_log_tile = self.skip_mode._nchw_to_nhwc(scale_log_tile.int())
                scale_log_tile = scale_log_tile.int()
                # skip_scale_log_tile = self.skip_mode._nchw_to_nhwc(mask_tile.bool())
                skip_scale_log_tile = mask_tile.bool()
                # residual_tile = self.skip_mode._nchw_to_nhwc(residual_tile.short())
                residual_tile = residual_tile.short()
                tile_control_points3.append(f'{TensorOps.get_hash(scale_log_tile.to(dtype=torch.int32))}')

                tile_control_points2.append(f'{TensorOps.get_hash(residual_tile[skip_scale_log_tile])}')
                tile_control_points5.append(f'{TensorOps.get_hash(mask_tile.to(dtype=torch.int32))}')
                #if tile_idx > 0:            # Test of streams decoding
                self._ac_encode_y(ec, residual_tile, scale_log_tile, skip_scale_log_tile, region_idx=tile_idx)
            self.logger.info(f"The first rvfull control point is {'-'.join(reversed(tile_control_points2))}\n")
            self.logger.info(f"The scale_log_masked_tile control point is {'-'.join(reversed(tile_control_points3))}\n")
            self.logger.info(f"The mask_tile control point is {'-'.join(reversed(tile_control_points5))}\n")
        else:
            # scale_log = self.skip_mode._nchw_to_nhwc(scale_log.int())
            # skip_scale_log = self.skip_mode._nchw_to_nhwc(self.skip_mode.mask().bool())
            # residual = self.skip_mode._nchw_to_nhwc(residual.short())
            # self._ac_encode_y(ec, residual, scale_log, skip_scale_log)
            # self.logger.info(f"The first control point is {TensorOps.get_hash(residual[skip_scale_log])}\n")

            # h_ls, w_ls = self.get_ls_shape(0)
            # z_s = residual.shape[1:3]
            # assert(h_ls==z_s[0] and w_ls==z_s[1])

            scale_log = scale_log.int()
            mask = self.skip_mode.mask()
            skip_scale_log = torch.ones_like(scale_log).bool() if mask is None else mask.bool()
            residual = residual.short()
            self._ac_encode_y(ec, residual, scale_log, skip_scale_log)
            self.logger.info(f"The quantized_residual control point is {TensorOps.get_hash(residual[skip_scale_log])}\n")
            self.logger.info(f"The quantized_residual2 control point is {TensorOps.get_hash(residual)}\n")
            h_ls, w_ls = self.get_ls_shape(0)
            z_s = residual.shape[2:]
            assert(h_ls==z_s[0] and w_ls==z_s[1])


    def _ac_encode_y(self, ec: ECModule, residual_hat: torch.Tensor, scale_log: torch.Tensor, masks: torch.Tensor, region_idx: int = 0):
        """run entropy model for latent space

        Args:
            ec (ECModule): entropy model.
            y_hat (torch.Tensor): output tensor of encoder in latent space.
            mean (torch.Tensor): mean value of probability model.
            scale_l (torch.Tensor): variance of left probability model.
            scale_r (torch.Tensor): variance of right probability model.
            weights (torch.Tensor): weights of probability model.
        """
        
        assert(scale_log.dtype == torch.int32)

        # self.logger.info(f"The first control point is {TensorOps.get_hash(residual_hat[masks])}\n")
        with self.set_ec_context(ec, "r", region_idx):
            ec.encode_sgt(residual_hat[:, :self.num_chs],
                        scale_log[:,:self.num_chs],
                        masks[:,:self.num_chs],
                        entropy_prob_model=self.entropy,
                        name=f'{self.owner.name} y_hat')


    def _ac_encode_z(self, ec: ECModule, z_hat, in_rdlr: bool = False) -> None:
        """run entropy model for hyper latent space

        Args:
            ec (ECModule): entropy model.
            z_hat (torch.Tensor): output tensor in hyper latent space.
        """
        x = z_hat + self.z_offset
        assert (x.min() >= 0) and (x.max() < self.z_range)

        if not in_rdlr:
            assert(x.dtype == torch.int8)

        with self.set_ec_context(ec, "z"):
            ec.encode_custom(x,
                         self.hyper_entropy,
                         self.z_range - 1,
                         self.z_offset,
                         name=f'{self.owner.name} z_hat')
        self.logger.info(f"The z_hat_0 control point is {TensorOps.get_hash(x)}\n")
        
    # ============================================================
    # Decode part

    @determinism
    def decode(self, ec: ECModule, decisions: Decisions = None) -> Decisions:
        if decisions is None:
            decisions = Decisions()
        h, w = self.get_processed_img_shape()
        down = int(math.log2(self.owner.alignment_size))
        latent_shape = torch.Size((1, self.chs_ls, math.ceil(h / (2 ** down)), math.ceil(w / (2 ** down))))
        self.tile_manager_hyper._init(latent_shape, self.owner.alignment_size)
        decisions.update(self.decode_z(ec, h, w))
        decisions.update(self.decode_y(ec, decisions, h, w))
        return decisions

    def decode_z(self, ec: ECModule, h: int, w: int) -> Decisions:
        """entropy decoder for hyper latent space

        Args:
            ec (ECModule): entropy model.
            h (int): height of the image.
            w (int): width of the image.
        Returns:
            (Decisions): a dict has latent tensor of z_hat.
        """
        disable_torch_random()
        disable_tf32()

        logger = self.logger

        self.params.load_params_from_owner(['code_mode'])
        
        h_ls, w_ls = self.get_ls_shape(1)

        z_hat_shape = (1, self.chs_ls, h_ls, w_ls)

        z_hat = self._ac_decode_z(ec, z_hat_shape, self.hyper_entropy)

        z_hat = z_hat.to(device=next(self.hyper_decoder.parameters()).device)

        ans = Decisions({'z_hat': z_hat})
        self.logger.info(f"The z_hat control point is {TensorOps.get_hash(z_hat)}\n")

        return ans


    def decode_y(self, ec: ECModule, decisions: Decisions, h: int, w: int) -> Decisions:
        """entropy decoder for latent space

        Args:
            ec (ECModule): entropy model.
            decisions (Decisions): a dict has results from hyper latent space.
            h (int): height of image.
            w (int): width of image.

        Returns:
            (Decisions): a dict has latent tensor and gain unit related parameters.
        """
        disable_torch_random()
        disable_tf32()
        logger = self.logger
        #present_substreams_list_Y = self.tile_manager_residual_bitstream.decoded_substreams_Y
        #present_substreams_list_UV = self.tile_manager_residual_bitstream.decoded_substreams_UV
        #subExtractionPossible = (self.tile_manager_residual_bitstream.region_residual_in_its_own_substream_flag == 1)
        #numTotalRegions = self.tile_manager_residual_bitstream.numHorRegions * self.tile_manager_residual_bitstream.numVerRegions
        #assert (subExtractionPossible) or (len(present_substreams_list_Y) == numTotalRegions), "Substream extraction only possible when region_residual_in_its_own_substream_flag flag is equal to 1"
        #assert (subExtractionPossible) or (len(present_substreams_list_UV) == numTotalRegions), "Substream extraction only possible when region_residual_in_its_own_substream_flag flag is equal to 1"
        self.params.load_params_from_owner(['code_mode'])
        decisions['quantizer'].update(self.quantizer.decode(ec, h=h, w=w))

        
        with self.get_profilers_ctx(f'{self.owner.name} hyper_decoder'):
            decisions['scale_log'] = self._decode_scale(decisions, h, w)

        decisions['skip_scale_log'] = decisions.get('scale_log')
        skip_scale_log = decisions['scale_log']
            
        decisions['quantizer'].update(self.quantizer.analyze(decisions, incl_list=['rvs']))       
        scale_log_new = self.quantizer.quantize_scale(skip_scale_log, decisions['quantizer'], incl_list=['rvs'])
        
        self.skip_mode._update_mask(skip_scale_log)      
        decisions['scale_log'] = scale_log_new
            
        rv_full = torch.zeros_like(scale_log_new)
        prev_module = self.context if self.use_context_module else self.hyper_decoder

        if self.tile_manager_residual_bitstream.enabled:
            self.logger.debug(
                f'Decode residual using these tiles: {repr(self.tile_manager_residual_bitstream.image_tiles_entropy_coding)}')

            tile_control_points2 = []
            tile_control_points3 = []
            tile_control_points5 = []
            for tile_idx, (image_tile_area, latent_tile_area) in enumerate(zip(self.tile_manager_residual_bitstream.image_tiles_entropy_coding,
                                            self.tile_manager_residual_bitstream.latent_tiles_entropy_coding)):

                tile_to_be_decoded = latent_tile_area


                mask_tile, output_shape = self.skip_mode.mask_with_tile(tile_to_be_decoded)
                # skip_scale_log_tile = self.skip_mode._nchw_to_nhwc(mask_tile.bool())
                skip_scale_log_tile = mask_tile.bool()
                scale_log_tile =  tiling.get_data(scale_log_new, tile_to_be_decoded)
                # scale_log_tile = self.skip_mode._nchw_to_nhwc(scale_log_tile.int())
                scale_log_tile = scale_log_tile.int()

                tile_control_points3.append(f'{TensorOps.get_hash(scale_log_tile.to(dtype=torch.int32))}')
                with self.get_profilers_ctx(f'{self.owner.name} _ac_decode_y'):
                    #if tile_idx in (present_substreams_list_Y if "model_y" in self.get_tool_url() else present_substreams_list_UV):
                    rv_tile = self._ac_decode_y(ec, scale_log_tile, skip_scale_log_tile, region_idx=tile_idx)
                    #else:
                    #    rv_tile = torch.zeros_like(scale_log_tile)

                tile_control_points2.append(f'{TensorOps.get_hash(rv_tile[skip_scale_log_tile.to(device=rv_tile.device)].to(dtype=torch.int16))}')
                prev_module = self.context if self.use_context_module else self.hyper_decoder
                # rv_tile = self.skip_mode._nhwc_to_nchw(rv_tile).to(device=next(prev_module.parameters()).device)
                rv_tile = rv_tile.to(device=next(prev_module.parameters()).device)

                tile_control_points5.append(f'{TensorOps.get_hash(mask_tile.to(dtype=torch.int32))}')
                check_pipelinig_support = True
                if check_pipelinig_support:
                    should_be_zeros = tiling.get_data(rv_full, tile_to_be_decoded)
                    assert should_be_zeros.abs().sum() == 0
                tiling.assign_data(rv_full, tile_to_be_decoded, rv_tile)


            # self.logger.info(f"The first control point is {'-'.join(tile_control_points)}\n")
            self.logger.info(f"The first rvfull control point is {'-'.join(tile_control_points2)}\n")
            self.logger.info(f"The scale_log_masked_tile control point is {'-'.join(tile_control_points3)}\n")
            self.logger.info(f"The mask_tile control point is {'-'.join(tile_control_points5)}\n")
        else:
            masks = self.skip_mode.mask().bool()
            scale_log_new = scale_log_new.int()

            with self.get_profilers_ctx(f'{self.owner.name} _ac_decode_y'):
                rv_full = self._ac_decode_y(ec, scale_log_new, masks)
            self.logger.info(f"The quantized_residual control point is {TensorOps.get_hash(rv_full[masks.to(device=rv_full.device)].to(dtype=torch.int16))}\n")
            self.logger.info(f"The quantized_residual2 control point is {TensorOps.get_hash(rv_full.to(dtype=torch.int16))}\n")
            prev_module = self.context if self.use_context_module else self.hyper_decoder
            rv_full = rv_full.to(device=next(prev_module.parameters()).device)

        decisions.update({'residual': rv_full.clone()})
        decisions.update({'residual_quant': rv_full.clone()})   
                
        rv_full_dequant = self.quantizer.dequantize_resi(rv_full, decisions['quantizer'])
        # self.logger.info(f"The second control point is {TensorOps.get_hash(rv_full_dequant)}\n")
        decisions.update({'residual': rv_full_dequant})         
        # self.logger.info(f"The dequantized_residual control point is {TensorOps.get_hash(rv_full[masks.to(device=rv_full.device)].to(dtype=torch.int16))}\n")
            
        return decisions


    def _decode_scale(self, decisions: Decisions, h: int, w: int) -> torch.Tensor:
        scale = self.hyper_scale_decoder(decisions['z_hat'], h, w)
        scale = self.quantizer.quantize_scale(scale, decisions['quantizer'], excl_list=['rvs'])
        return scale

    def _ac_decode_z(self, ec: ECModule, z_hat_shape, hyper_entropy) -> torch.Tensor:
        """arithmetic decode for hyper latent space

        Args:
            ec (ECModule): entropy model.
            z_hat_shape (tuple): size of the tensor in hyper latent space.
            hyper_entropy (FactorizedProbModel): factorized entropy model.
        Returns:
            z_hat (torch.Tensor): reconstructed hyper latent tensor.
        """
        (_, ZC, ZH, ZW) = z_hat_shape
        symbols = torch.arange(self.z_range, device=next(hyper_entropy.parameters()).device).view(
            1, 1, self.z_range)
        symbols = symbols.repeat(ZC, 1, 1).float()

        with self.get_profilers_ctx('z arithmetic decode api'), self.set_ec_context(ec, "z"):
            z_hat = ec.decode_custom(z_hat_shape,
                                     hyper_entropy,
                                     self.z_range - 1,
                                     self.z_offset,
                                     name=f'{self.owner.name} z_hat')
            
        self.logger.info(f"The z_hat_0 control point is {TensorOps.get_hash(z_hat)}\n")

        assert(z_hat.dtype == torch.int8)
        z_hat = z_hat.to(next(hyper_entropy.parameters()).device) - self.z_offset

        return z_hat

    def _cal_step_size(self, lh, lw):
        num_threads = self.get_owner_param('num_threads_r')
        delta_num_elements = num_threads * 4 # number of elements of each layer should be multiple of 4 * num_ans_threads
        step_size = max(min(self.num_decode_chs, self.num_chs), 1)
        for item in range(1, min(self.num_decode_chs, self.num_chs) + 1):
            if int(lh * lw * item) % delta_num_elements == 0:
                step_size = item
                break
        return step_size

    def _ac_decode_y(self, ec: ECModule, scale_log: torch.Tensor, masks: torch.Tensor, region_idx: int = 0) -> torch.Tensor:
        """Select to run arithmetic decoder for full image or patches
        Args:
            ec (ECModule): entropy model.
            y_hat_shape (torch.Size): size of latent tensor.
            psi (torch.Tensor): tensor psi for entropy model.

        """
        assert(scale_log.dtype == torch.int32)

        self.update_entropy_model()

        with self.set_ec_context(ec, "r", region_idx):
            #if self.num_decode_chs is not None and self.num_decode_chs >= 0:
            step_size = self._cal_step_size(lh=scale_log.shape[2], lw=scale_log.shape[3])
            rv = torch.zeros(scale_log.shape, device=scale_log.device, dtype=torch.float32)
            for item in range(0, min(self.num_decode_chs, self.num_chs), step_size):
                id_layer0 = item
                id_layer1 = min(id_layer0 + step_size, self.num_chs)
                scale_log2dec = scale_log[:, id_layer0:id_layer1, :, :]
                masks2dec = masks[:, id_layer0:id_layer1, :, :]
                rv_from_dec = ec.decode_sgt(scale_log2dec, masks2dec, entropy_prob_model=self.entropy, name=f'{self.owner.name} y_hat')
                rv[:, id_layer0:id_layer1, :, :] = rv_from_dec
            #else:
            #    rv = ec.decode_sgt(scale_log, masks, entropy_prob_model=self.entropy, name=f'{self.owner.name} y_hat')
        # self.logger.info(f"The first control point is {TensorOps.get_hash(rv[masks].to(dtype=torch.int16))}\n")

        return rv

    def export_models(self, output_dir: str, opset_version: int):
        from src.codec.common import ModulesContext
        from src.codec.components.base_layers import Conv2di
        output_dir = os.path.join(output_dir, self.name)
        os.makedirs(output_dir, exist_ok=True)
        #torch.onnx.export(self.hyper_entropy, 
        #                  torch.rand([1, self.chs_ls, 256, 256], device=self.device), 
        #                  os.path.join(output_dir, "hyper_entropy.onnx"),
        #                  export_params = True, opset_version = opset_version,
        #                  #input_names=['z'], output_names=['p'],
        #                  #dynamic_axes={'z': [2,3], 'p': [2,3]}
        #                  )
        ## Export the current frequency table
        #cdfs_z_tensor = self.hyper_entropy.get_freq_table(self.chs_ls, self.z_range, self.z_offset)
        #np.savetxt(os.path.join(output_dir, "cdf_z_table.csv"), cdfs_z_tensor.cpu().numpy().astype(np.int), delimiter=',', fmt='%d')
        ## Export hyper decoder
        torch.onnx.export(self.hyper_decoder, 
                          torch.rand([1, self.chs_ls, 256, 256], device=self.device), 
                          os.path.join(output_dir, "hyper_decoder.onnx"),
                          export_params = True, opset_version = opset_version,
                          input_names=['z'], output_names=['p'],
                          dynamic_axes={'z': [2,3], 'p': [2,3]}
                          )
        ## Export hyper encoder
        torch.onnx.export(self.hyper_encoder, 
                          torch.rand([1, self.chs_ls, 256, 256], device=self.device), 
                          os.path.join(output_dir, "hyper_encoder.onnx"),
                          export_params = True, opset_version = opset_version,
                          input_names=['y'], output_names=['z'],
                          dynamic_axes={'y': [2,3], 'z': [2,3]}
                          )
        ## Export hyper scale decoder
        #with ModulesContext({"is_quantized": torch.tensor([False], device=self.device)}, self.hyper_scale_decoder, Conv2di):
        torch.onnx.export(self.hyper_scale_decoder, 
                            torch.randint(0, 20, [1, self.chs_ls, 256, 256], device=self.device), 
                            os.path.join(output_dir, "hyper_scale_decoder.onnx"),
                            export_params = True, opset_version = opset_version,
                            input_names=['z'], output_names=['I_s'],
                            dynamic_axes={'z': [2,3], 'I_s': [2,3]}
                            )
        if self.use_context_module:
            self.context.export_models(os.path.join(output_dir, "MCM"), opset_version)
      