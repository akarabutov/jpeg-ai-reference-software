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

from builtins import staticmethod

from typing import List, Tuple
import math
import numpy as np

import torch
import re
import torch.nn.functional as F
from collections import OrderedDict

from src.codec.coding_tools.interfaces import ParamsCommonObj, AttrsProxy
from src.codec.common import (Decisions, Image, tiling, determinism, disable_tf32, disable_torch_random, TensorOps)
from src.codec.entropy_coding import ECModule, HeaderCoder
from src.codec.coding_tools.quantization import Quantizer
from src.codec.coding_tools.tiling import TileManager

from ...interfaces import HeaderProxy
from ..base import CoreModelBase
##
from .sep_chan_tool import SepChannelsSGMMTool
from .params import CcsCommonParams
from copy import deepcopy


from src.codec.coding_tools.quality_map import QualityMap

class CcsGvaeSGMM(CoreModelBase):
    """Base model for image compression
    """

    def __init__(self, **kwards):
        super(CcsGvaeSGMM, self).__init__(**kwards, stream_header_part='pic_header')
        self._params_ccs_model = CcsCommonParams()
        N_luma = 160
        N_chroma = 96
        model_y = SepChannelsSGMMTool(1, 
                                           chs_ls=N_luma, 
                                           ccs_id=0, 
                                           stream_base_comp=0,
                                           stream_header_part="pic_header"
                                           )
        model_uv = SepChannelsSGMMTool(2, 
                                            chs_ls=N_chroma, 
                                            chs_ls_supp=N_luma, 
                                            chs_in_supp=1, 
                                            downsample_factor=2, 
                                            ccs_id=1, 
                                            stream_base_comp=1,
                                            stream_header_part="pic_header"
                                            )
        self.models_list = [model_y, model_uv]
        self.default_model_name = ''

        self.model_common = ParamsCommonObj(self.models_list)
        self.model_y = model_y
        self.model_uv = model_uv
                      
        self.qual_map = QualityMap(stream_header_part="pic_header",
                                   enable_flag_name='gain_3D_enable_flag')
               
        self._quantizers_proxy = HeaderProxy(self.models_list, Quantizer, False)
        self.McmOverlap = 0
        self.HyperDecoderOverlap = 0
                
    def get_sgm_entropy_model(self):
        return self.model_y.get_sgm_entropy_model()
    
    def get_ls_shape(self, level: int = 0) -> torch.Size:
        return self.model_y.get_ls_shape(level)

        
    def get_internal_data_range(self) -> List[int]:
        """Return range of internal data representation, i.e. [0,1]

        Returns:
            (list): range of data presentation
        """
        return self.model_y.internal_data_range

    def get_alignment_size(self) -> int:
        """Get maximum alignment of models

        Returns:
            (int): maximum alignment of models
        """
        return max(self.model_y.get_alignment_size(), self.model_uv.get_alignment_size())

    @staticmethod
    def calc_downsampled_shape(*args, scale=2) -> List[int]:
        """Calculate size of downsampled image

        Args:
            scale (int, optional): Scale of downsampling. Defaults to 2.

        Returns:
            List[int]: size of downsampled image
        """
        return list(Image.scale_size(list(args), 1 / float(scale)))
    
    def _params_loaded(self) -> None:
        self.McmOverlap = self.mcm_overlap_in_latent_samples * 4
        self.HyperDecoderOverlap = self.hyper_decoder_overlap_in_latent_samples * 2

    def synthesisTileToRegion(self,model, tileIdx, full_img_height, full_img_width):
            if model.common_modules.tile_manager_hyper.region_residual_in_its_own_substream_flag == 0:
                return 0
            numHorRegions = model.common_modules.tile_manager_hyper.numHorRegions
            numVerRegions = model.common_modules.tile_manager_hyper.numVerRegions
            verRegionSize = int(math.floor(math.floor((full_img_height + 127) / 128) / numVerRegions) * 128)
            horRegionSize = int(math.floor(math.floor((full_img_width + 127) / 128) / numHorRegions) * 128)
            tile_size = model.tile_manager_synthesis.tile_size
            if tile_size == 0:
                tile_size = int(math.sqrt(model.tile_manager_synthesis.numSamplesPerTile))
            tile_idx = -1
            for i  in range (numHorRegions):
                verSize =  verRegionSize if i < numHorRegions-1 else full_img_height - i*verRegionSize
                for l in range(0,verSize, tile_size):
                    for j in range(numVerRegions):
                        horSize =  horRegionSize if j < numVerRegions-1 else full_img_width - j*horRegionSize
                        for k in range(0,horSize, tile_size):
                            tile_idx+=1
                            if tile_idx == tileIdx:
                                return i*numVerRegions + j
                            
    def calc_numHor_numVer_regions(self, height, width):
        if self.NumSamplesInRegion > 0:
            if self.region_partitioning_flag == 0 or height * width <= self.NumSamplesInRegion:
                self.numHorRegions, self.numVerRegions = 1, 1
                self.region_partitioning_flag = 0
                self.region_residual_in_its_own_substream_flag = 0
            else:
                step = int(math.sqrt(self.NumSamplesInRegion))
                max_ver_num = math.ceil(height / 256)
                max_hor_num = math.ceil(width / 512)
                self.numHorRegions = max(1, min(math.floor(width / step), max_hor_num))
                self.numVerRegions = max(1, min(math.floor(height / step), max_ver_num))
                self.region_partitioning_flag = 1
        self.set_ec_params()

    def setup_dec_tile_managers_of_model(self, model: SepChannelsSGMMTool) -> TileManager:
        full_img_height, full_img_width = self.get_original_img_shape()
        #self.model_y.get_processed_img_shape()
        img_height, img_width = model.get_processed_img_shape()
        # coeff = model.alignment_size_dec // model.alignment_size
        img_shape = (1,1,img_height, img_width)
        full_img_shape = (1,1,full_img_height, full_img_width)
        # out_img_shape = (1,1,img_height*coeff, img_width*coeff)
        latent_y_height = math.ceil(img_height / model.alignment_size)
        latent_y_width = math.ceil(img_width / model.alignment_size)
        latent_y_shape = (1, model.chs_ls, latent_y_height, latent_y_width)
        latent_psi_height = math.ceil(img_height / (model.alignment_size * 2 ))
        latent_psi_width = math.ceil(img_width / (model.alignment_size * 2 ))
        latent_psi_shape = (1, model.chs_ls*4, latent_psi_height, latent_psi_width)
        latent_z_height = math.ceil(img_height / (model.alignment_size * 4 ))
        latent_z_width = math.ceil(img_width / (model.alignment_size * 4 ))
        latent_z_shape = (1, model.chs_ls*4, latent_z_height, latent_z_width)


        # this is weird. why in common modules?1
        model.common_modules.tile_manager_hyper._init(latent_y_shape, model.alignment_size)
        model.get_base_tile_params(model.common_modules.tile_manager_residual_bitstream)
        model.common_modules.tile_manager_residual_bitstream.setup_tiles_dec_from_region_tile_manager(model.common_modules.tile_manager_hyper, full_img_shape, latent_y_shape, latent_psi_shape, latent_z_shape, alighment_size=model.alignment_size)

        use_region_tile_manager_also_for_hd_mcm = True

        if use_region_tile_manager_also_for_hd_mcm:
            model.get_base_tile_params(model.common_modules.tile_manager_hd)
            model.get_base_tile_params(model.common_modules.tile_manager_mcm)
            model.common_modules.tile_manager_hd.setup_tiles_dec_from_region_tile_manager(model.common_modules.tile_manager_hyper, full_img_shape, latent_y_shape, latent_psi_shape, latent_z_shape, alighment_size=model.alignment_size)
            model.common_modules.tile_manager_mcm.setup_tiles_dec_from_region_tile_manager(model.common_modules.tile_manager_hyper, full_img_shape, latent_y_shape, latent_psi_shape, latent_z_shape, alighment_size=model.alignment_size)
        else:
            model.common_modules.tile_manager_hd.setup_tiles_enc(full_img_shape, latent_y_shape, latent_psi_shape, latent_z_shape)
            model.common_modules.tile_manager_mcm.setup_tiles_enc(full_img_shape, latent_y_shape, latent_psi_shape, latent_z_shape)
        model.get_base_tile_params(model.tile_manager_synthesis)
        model.tile_manager_synthesis.setup_tiles_dec(full_img_shape, latent_y_shape, latent_psi_shape, latent_z_shape)
    
    def compress(self, img: Image, decision: Decisions = {}) -> Decisions:
        """Compress input image to decisions

        Args:
            img (Image): input image.
            decision (Decisions, optional): preset decisions. Defaults to {}.

        Returns:
            decisions (Decisions): a dict with latent and hyper latent tensors
        """
        clip_min, clip_max = self.BDL_clipping_range #here add cliping
        self.model_y.beta_displacement_log = np.clip(self.owner.beta_displacement_log_Y, clip_min, clip_max)
        self.model_uv.beta_displacement_log = np.clip(self.owner.beta_displacement_log_UV, clip_min, clip_max)
        
        c_ver = self.get_owner_param('c_ver')
        c_hor = self.get_owner_param('c_hor')
        img.to_YUV_()
        img.convert_range_(self.get_internal_data_range())
        ############### for RGB input
        h, w = img.shape[-2:]
        img.pad_(w % 2, h % 2, mode='replicate', comp_list=['a'])
        luma = img.get_component('a')
        chroma_sup_info = F.pixel_unshuffle(luma, 2)
        chrom_subsmpl_fmt = Image.get_format_from_subsampling(c_ver, c_hor)
        img.to_format_(chrom_subsmpl_fmt)

        if (c_ver==1 and c_hor==1):  # 444
            img.pad_(w % 2, h % 2, mode='replicate', comp_list=['b', 'c'])
            chroma = F.pixel_unshuffle(torch.cat((img.get_component('b'),img.get_component('c')), dim=1), 2)
        elif (c_ver==2 and c_hor==2):  # 420 
            chroma = torch.cat((img.get_component('b').repeat(1, 4, 1, 1),img.get_component('c').repeat(1, 4, 1, 1)), dim=1)
        elif (c_ver==1 and c_hor==2):
            # 422 case
            img.pad_(0, h%2, mode='replicate', comp_list=['b', 'c'])
            c1 = img.get_component('b')[:,:,::2,:].repeat(1, 2, 1, 1)
            c2 = img.get_component('b')[:,:,1::2,:].repeat(1, 2, 1, 1)
            c3 = img.get_component('c')[:,:,::2,:].repeat(1, 2, 1, 1)
            c4 = img.get_component('c')[:,:,1::2,:].repeat(1, 2, 1, 1)
            chroma = torch.cat((c1, c2, c3, c4), dim=1)
        else:
            raise NotImplementedError
        chroma = torch.cat( (chroma_sup_info, chroma), dim=1 )        
        if 'model_y' not in decision and 'model_uv' not in decision:
            ans = Decisions()
            full_img_height,full_img_width = img.shape[-2:]
            self.calc_numHor_numVer_regions(full_img_height,full_img_width)
            
            self.get_profilers().start('CCS Y compress')
            ans['model_y'] = self.model_y.compress(luma)
            self.get_profilers().finish('CCS Y compress')
            chroma_decisions = Decisions()
            # Put qp_map from decision['model_y'] to chroma_decisions
            if 'qual_map' in ans['model_y']['quantizer']:
                chroma_decisions.update({'quantizer':{'qual_map':ans['model_y']['quantizer']['qual_map']}})
            
            
            self.get_profilers().start('CCS UV compress')
            ans['model_uv'] = self.model_uv.compress(chroma, decision=chroma_decisions)
            #ans['chroma_shape'] = chroma.shape
            self.get_profilers().finish('CCS UV compress')
            return ans            
        else:            
            self.get_profilers().start('CCS Y compress')
            decision['model_y'] = self.model_y.compress(luma.clone(), decision['model_y'])
            self.get_profilers().finish('CCS Y compress')
            self.get_profilers().start('CCS UV compress')
            decision['model_uv'] = self.model_uv.compress(chroma.clone(), decision['model_uv'])
            self.get_profilers().finish('CCS UV compress')
            return decision
            
            
    def decompress(self, decisions: Decisions, *args, **kwargs) -> Image:
        """decompress function will generate reconstructed image

        Args:
            decisions (Decisions): a dict with latent and hyper latent information.
        Returns:
            rec (Image): reconstructed image.
        """
        rec_Y, rec_UV = self.forward(decisions, *args, **kwargs)
        rec_U = rec_UV[:, 0:1]
        rec_V = rec_UV[:, 1:2]
        c_ver = self.get_owner_param('c_ver')
        c_hor = self.get_owner_param('c_hor')
        s_ver = self.get_owner_param('s_ver')
        s_hor = self.get_owner_param('s_hor')

        img_fmt = Image.get_format_from_subsampling(c_ver, c_hor)
        ans = Image.create_from_tensors(rec_Y,
                                        rec_U,
                                        rec_V,
                                        self.get_internal_data_range(),
                                        format=img_fmt,
                                        color_space='yuv',
                                        bit_depth=self.get_owner_param('image_data_bits'))
        out_img_fmt = Image.get_format_from_subsampling(s_ver, s_hor)
        ans.to_format_(out_img_fmt)

        return ans
    

        
    
    def forward(self, decisions: Decisions, *args, **kwargs):
        h, w = self.get_processed_img_shape()
        c_ver = self.get_owner_param('c_ver')
        c_hor = self.get_owner_param('c_hor')

        #subExtractionPossible = (self.model_y.tile_manager_mcm.region_residual_in_its_own_substream_flag == 1)
        #numTotalRegions = self.model_y.tile_manager_mcm.numHorRegions * self.model_y.tile_manager_mcm.numVerRegions
        #assert (subExtractionPossible) or (len(present_substreams_list_Y) == numTotalRegions), "Substream extraction only possible when region_residual_in_its_own_substream_flag flag is equal to 1"
        #assert (subExtractionPossible) or (len(present_substreams_list_UV) == numTotalRegions), "Substream extraction only possible when region_residual_in_its_own_substream_flag flag is equal to 1"
        diff_display_img_width = self.get_owner_param('diff_display_img_width')
        diff_display_img_height = self.get_owner_param('diff_display_img_height')

        self.setup_dec_tile_managers_of_model(self.model_y)
        self.setup_dec_tile_managers_of_model(self.model_uv)

        # # this is weird. why in common modules?
        # self.model_y.common_modules.tile_manager_hyper._init(self.model_y.tile_manager_synthesis.latent_shape )

        rec_Y = torch.zeros((1,1,h,w), device=self.device, dtype=torch.float)
        rec_UV = torch.zeros((1,2,h,w), device=self.device, dtype=torch.float)


        self.logger.debug(
            f'Hyper decoder using these tiles for Luma on image level: {repr(self.model_y.common_modules.tile_manager_hd.image_tiles)}, psi: {repr(self.model_y.common_modules.tile_manager_hd.latent_tiles_psi)}, Z-hat: {repr(self.model_y.common_modules.tile_manager_hd.latent_tiles_z)}')
        self.logger.debug(
            f'Hyper decoder using these tiles for Chroma on image level: {repr(self.model_uv.common_modules.tile_manager_hd.image_tiles)}, psi: {repr(self.model_uv.common_modules.tile_manager_hd.latent_tiles_psi)}, Z-hat: {repr(self.model_uv.common_modules.tile_manager_hd.latent_tiles_z)}')
        self.logger.debug(
            f'Context using these tiles for Luma on image level: {repr(self.model_y.common_modules.tile_manager_mcm.image_tiles)}, Y-hat: {repr(self.model_y.common_modules.tile_manager_mcm.latent_tiles)}, psi: {repr(self.model_y.common_modules.tile_manager_mcm.latent_tiles_psi)}')
        self.logger.debug(
            f'Context using these tiles for Chroma on image level: {repr(self.model_uv.common_modules.tile_manager_mcm.image_tiles)}, Y-hat: {repr(self.model_uv.common_modules.tile_manager_mcm.latent_tiles)}, psi: {repr(self.model_uv.common_modules.tile_manager_mcm.latent_tiles_psi)}')

        self.logger.debug(
            f'Synthesis using these tiles for Luma on image level: {repr(self.model_y.tile_manager_synthesis.image_tiles)}, Y-hat: {repr(self.model_y.tile_manager_synthesis.latent_tiles)}')
        self.logger.debug(
            f'Synthesis using these tiles for Chroma on image level: {repr(self.model_uv.tile_manager_synthesis.image_tiles)}, Y-hat: {repr(self.model_uv.tile_manager_synthesis.latent_tiles)}')

        control_points_Y = {}
        control_points_UV = {}

        # can only iterate over zip once, need make anew. also use decoder tiles from here
        zipped_tiles_descriptions = zip(
            self.model_y.common_modules.tile_manager_mcm.latent_tiles,
            self.model_uv.common_modules.tile_manager_mcm.latent_tiles,
            )


        self.model_y.logger.info(f"The residual control point is {TensorOps.get_hash(decisions['model_y']['residual'])}\n")
        self.model_uv.logger.info(f"The residual control point is {TensorOps.get_hash(decisions['model_uv']['residual'])}\n")

        # get residual tiles
        decisions['model_y']['residual_dq_tiles'] = dict()
        decisions['model_uv']['residual_dq_tiles'] = dict()
        for tile_descriptions in zipped_tiles_descriptions:
            (
                Y_y_tile_desc, 
                UV_y_tile_desc, 
            ) = tile_descriptions

            decisions['model_y']['residual_dq_tiles'][Y_y_tile_desc] = tiling.get_data(decisions['model_y']['residual'], Y_y_tile_desc)
            decisions['model_uv']['residual_dq_tiles'][UV_y_tile_desc] = tiling.get_data(decisions['model_uv']['residual'], UV_y_tile_desc)


        control_points_Y['residual_dq_tiles'] = []
        for Y_y_tile_desc in sorted(decisions['model_y']['residual_dq_tiles'].keys()):
            hash_residual = TensorOps.get_hash(decisions['model_y']['residual_dq_tiles'][Y_y_tile_desc])
            control_points_Y['residual_dq_tiles'].append(f'{hash_residual}')
        self.model_y.logger.info(f"The residual_tiles control point is {'-'.join(control_points_Y['residual_dq_tiles'])}\n")

        control_points_UV['residual_dq_tiles'] = []
        for UV_y_tile_desc in sorted(decisions['model_uv']['residual_dq_tiles'].keys()):
            hash_residual = TensorOps.get_hash(decisions['model_uv']['residual_dq_tiles'][UV_y_tile_desc])
            control_points_UV['residual_dq_tiles'].append(f'{hash_residual}')
        self.model_uv.logger.info(f"The residual_tiles control point is {'-'.join(control_points_UV['residual_dq_tiles'])}\n")


        decisions['model_y']['tiles_hd'] =  Decisions()
        decisions['model_uv']['tiles_hd'] =  Decisions()
        decisions['model_y']['tiles_mcm_resi'] =  Decisions()
        decisions['model_uv']['tiles_mcm_resi'] =  Decisions()
        decisions['model_y']['tiles_synthesis'] = Decisions()
        decisions['model_uv']['tiles_synthesis'] = Decisions()

        self.get_profilers().start('CCS Y hyper_decoder')
        for tile_info in tiling.ColocatedTiles.iter_colocated_grids(self.model_y.common_modules.tile_manager_hd):
            self.model_y.common_modules.hyper_decode_tile(
                decisions['model_y'], 
                tile_info,
                tile_info.img)
        self.get_profilers().finish('CCS Y hyper_decoder')

        self.get_profilers().start('CCS UV hyper_decoder')
        for tile_info in tiling.ColocatedTiles.iter_colocated_grids(self.model_uv.common_modules.tile_manager_hd):
            self.model_uv.common_modules.hyper_decode_tile(
                decisions['model_uv'], 
                tile_info,
                tile_info.img.downscale(2))
        self.get_profilers().finish('CCS UV hyper_decoder')

        # would not needed when merging hd and MCM use same tiles
        # combine y_hat overlapping areas, replacing overlaps outside of core region of a tile with part of neighboring tile
        decisions['model_y']['psi'] = self.model_y.common_modules.merge_psi_overlaps_of_tiles(decisions['model_y'])
        decisions['model_uv']['psi'] = self.model_uv.common_modules.merge_psi_overlaps_of_tiles(decisions['model_uv'])
        self.model_y.common_modules.extract_psi_for_mcm(decisions['model_y'], decisions['model_y']['psi'])
        self.model_uv.common_modules.extract_psi_for_mcm(decisions['model_uv'], decisions['model_uv']['psi'])


        self.get_profilers().start('CCS Y decompress_ar_scale')
        for region_idx, tile_info in enumerate(tiling.ColocatedTiles.iter_colocated_grids(self.model_y.common_modules.tile_manager_mcm)):
            #if region_idx in present_substreams_list_Y:
            self.model_y.common_modules.decompress_ar_scale_tile(
                decisions['model_y'], 
                tile_info)
        self.get_profilers().finish('CCS Y decompress_ar_scale')

        self.get_profilers().start('CCS UV decompress_ar_scale')
        for region_idx, tile_info in enumerate(tiling.ColocatedTiles.iter_colocated_grids(self.model_uv.common_modules.tile_manager_mcm)):
            #if region_idx in present_substreams_list_UV:
            self.model_uv.common_modules.decompress_ar_scale_tile(
                decisions['model_uv'], 
                tile_info)
        self.get_profilers().finish('CCS UV decompress_ar_scale')

        self.model_y.common_modules.print_control_points_y_hat_psi(decisions['model_y'])
        self.model_uv.common_modules.print_control_points_y_hat_psi(decisions['model_uv'])

        # combine y_hat overlapping areas, replacing overlaps outside of core region of a tile with part of neighboring tile
        # 1. combine y_hat
        decisions['model_y']['y_hat'] = self.model_y.common_modules.merge_y_hat_overlaps_of_tiles(decisions['model_y'])
        decisions['model_uv']['y_hat'] = self.model_uv.common_modules.merge_y_hat_overlaps_of_tiles(decisions['model_uv'])

        decisions['model_y']['residual']
        # 2. postprocessing
        with self.model_y.get_profilers_ctx(f'{self.model_y.name} ls_postprocessing'):
            self.model_y.ls_processing.post_processing(decisions['model_y'])      
        with self.model_uv.get_profilers_ctx(f'{self.model_uv.name} ls_postprocessing'):
            self.model_uv.ls_processing.post_processing(decisions['model_uv'])      

        # 3. get y_hat synthesis tiles
        self.model_y.common_modules.extract_y_hat_for_synthesis_tiles(decisions['model_y'], decisions['model_y']['y_hat'])
        self.model_uv.common_modules.extract_y_hat_for_synthesis_tiles(decisions['model_uv'], decisions['model_uv']['y_hat'])

        self.get_profilers().start('CCS Y decompress')
        for tile_idx, tile_info in enumerate(tiling.ColocatedTiles.iter_colocated_grids(self.model_y.tile_manager_synthesis)):
            region_idx = self.synthesisTileToRegion(self.model_y, tile_idx, h, w)
            #if region_idx in present_substreams_list_Y:
            self.model_y.common_modules.decompress_y_hat_to_image_tile(
                decisions['model_y'],
                tile_info,
                rec_Y)
        self.get_profilers().finish('CCS Y decompress')

        self.get_profilers().start('CCS UV decompress')
        for tile_idx, tile_info in enumerate(tiling.ColocatedTiles.iter_colocated_grids(self.model_uv.tile_manager_synthesis)):
            region_idx = self.synthesisTileToRegion(self.model_y, tile_idx, h, w)
            # TODO: this will only work if luma and chroma tiles are aligned. Add check if they are and if not extract correct region of y_hat from luma
            #if region_idx in present_substreams_list_UV:
            supp_info_uv = decisions['model_y']['tiles_synthesis'][tile_info.img]['y_hat']
            self.model_uv.common_modules.decompress_y_hat_to_image_tile(
                decisions['model_uv'],
                tile_info,
                rec_UV,
                supp_info_uv)
        self.get_profilers().finish('CCS UV decompress')

        output_width = w - diff_display_img_width
        output_height = h - diff_display_img_height
        rec_Y = rec_Y[:,:,:output_height,:output_width]
        rec_UV = rec_UV[:,:,:output_height:c_ver, :output_width:c_hor]
        return rec_Y, rec_UV

    def encode(self, ec: ECModule, decisions: Decisions, h: int = None, w: int = None):
        """Encode decisions to bitstream

        Args:
            ec (ECModule): entropy coder
            decisions (Decisions): decisions which will be encoded
            h (int, optional): height of image if it's not the same as get_processed_img_shape() provides. Defaults to None.
            w (int, optional): width of image if it's not the same as get_processed_img_shape() provides. Defaults to None.
        """
        
        rev_order = ec.bs.reverse_encode_order
        models_list = reversed(self.models_list) if rev_order else self.models_list
        
        for m in models_list:
            d = decisions.get(m.name, None)
            if d is not None:
                m.encode(ec, d, h, w, rev_order=rev_order)
        # Encode  Luma decisions
        if 'qual_map' in decisions['model_y']['quantizer']:
            self.qual_map.encode(ec, decisions['model_y']['quantizer']['qual_map'], h, w)

    def decode(self, ec: ECModule) -> Decisions:
        """Decode decisions from bitstream

        Args:
            ec (ECModule): entropy coder

        Returns:
            Decisions: output decisions
        """

        self.setup_dec_tile_managers_of_model(self.model_y)
        self.setup_dec_tile_managers_of_model(self.model_uv)

        ans = Decisions()
        
        qual_map_value = self.qual_map.decode(ec)
        # Put to decisions of Luma and Chroma
        # qual_map = {'quantizer':{'qual_map': self.qual_map.decode(ec)}}
        # ans.update({'model_uv':{'quantizer':{'qual_map': qual_map}}})
        
        for m in self.models_list:
            qual_map = Decisions()
            qual_map.update({'quantizer':{'qual_map': qual_map_value}})
            ans[m.name] = m.decode(ec,qual_map)
        # Put to decisions of Luma and Chroma
        
        return ans

    def decode_header(self, ec: HeaderCoder) -> None:
        """Decode signalling type for tiling

        Args:
            ec (HeaderCoder): binary coder
        """
        multi_threading_z = ec.decode([1], bits_count=1, name='multi_threading_z').item()
        if multi_threading_z:
            log2_num_threads_z_minus1 = ec.decode([1], bits_count=2, name='log2_num_threads_z_minus1').item()
            self.num_threads_z = 1 << (log2_num_threads_z_minus1+1)
        else:
            self.num_threads_z = 1
            
        self._quantizers_proxy.decode_headers(ec, True)
        
        self.region_partitioning_flag = int(ec.decode([1], 1, name='region_partitioning_flag').item())
        if self.region_partitioning_flag:
            self.numVerRegions = int(ec.decode([1], bits_count=7, name='num_ver_splits_minus1').item()) + 1
            self.numHorRegions = int(ec.decode([1], bits_count=7, name='num_hor_splits_minus1').item()) + 1
            max_ver_num = math.ceil(self.get_owner_param('img_height') / 256)
            max_hor_num = math.ceil(self.get_owner_param('img_width') / 512)
            assert self.numVerRegions <= max_ver_num, "Incorrect number of vertical regions"
            assert self.numHorRegions <= max_hor_num, "Incorrect number of horizontal regions"
            self.region_residual_in_its_own_substream_flag = int(
                ec.decode([1], 1, name='region_residual_in_its_own_substream_flag').item())
            if self.region_residual_in_its_own_substream_flag == 0:
                self.hyper_decoder_overlap_in_latent_samples = int(ec.decode([1], bits_count=2, name='hyper_decoder_overlap_in_latent_samples').item())
                self.HyperDecoderOverlap = self.hyper_decoder_overlap_in_latent_samples * 2
                self.mcm_overlap_in_latent_samples = int(ec.decode([1], bits_count=4, name='mcm_overlap_in_latent_samples').item()) 
                self.McmOverlap = self.mcm_overlap_in_latent_samples * 4

        self._quantizers_proxy.decode_headers(ec, False)
                
        # Pass parameters to bitstream structure
        self.set_ec_params()
        
        

    def encode_header(self, ec: HeaderCoder) -> None:
        """Encode signalling type for tiling

        Args:
            ec (HeaderCoder): binary coder
        """
        ec.encode(self.num_threads_z > 1, max_symbol_value=1, name='multi_threading_z')
        if self.num_threads_z > 1:
            tmp = math.log2(self.num_threads_z)
            assert abs(int(tmp) - tmp) < 1E-5
            log2_num_threads_z_minus1 = int(tmp) - 1
            ec.encode(log2_num_threads_z_minus1, bits_count=2, name='log2_num_threads_z_minus1')
            
        self._quantizers_proxy.encode_headers(ec, True)

        ec.encode(self.region_partitioning_flag, 1, name='region_partitioning_flag')
        if self.region_partitioning_flag:
            ec.encode(self.numVerRegions - 1, bits_count=7, name='num_ver_splits_minus1')
            ec.encode(self.numHorRegions - 1, bits_count=7, name='num_hor_splits_minus1')
            ec.encode(self.region_residual_in_its_own_substream_flag, 1,
                name="region_residual_in_its_own_substream_flag")
            if self.region_residual_in_its_own_substream_flag == 0:
                #assert (self.hyper_decoder_overlap_in_latent_samples & 0x1) == 0
                #assert (self.mcm_overlap_in_latent_samples & 0x1) == 0
                ec.encode(self.hyper_decoder_overlap_in_latent_samples, bits_count=2,
                        name="hyper_decoder_overlap_in_latent_samples")
                ec.encode(self.mcm_overlap_in_latent_samples, bits_count=4,
                        name="mcm_overlap_in_latent_samples")
                
        self._quantizers_proxy.encode_headers(ec, False)

                   

    def set_ec_params(self) -> None:
        EC = self.get_object_by_url('ce.EC')
        setattr(EC, 'region_residual_in_its_own_substream_flag', self.region_residual_in_its_own_substream_flag)
        setattr(EC, 'num_regions', self.numHorRegions * self.numVerRegions)
        setattr(EC, 'num_threads_z', self.num_threads_z)
        