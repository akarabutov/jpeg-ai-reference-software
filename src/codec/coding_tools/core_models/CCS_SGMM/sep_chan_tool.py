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
import math
from typing import List
import torch

from src.codec.common import (Decisions, determinism, tiling, Image, TensorOps)
from src.codec.components import get_size_on_depth

from src.codec.coding_tools.components_wrappers import create_datadecoder_instance, create_dataencoder_instance
from src.codec.coding_tools.interfaces import AttrsProxy, ToolEngine
from src.codec.coding_tools.tiling import TileManager, TileManagerHyper
from src.codec.coding_tools.ls_processing import LSProcessing
##
from  src.codec.components.contexts.utils import Upsample_proc
from src.codec.entropy_coding.header_module import HeaderCoder
from .params import (CcsSepChannelParams, CcsSingleModelParams)
from .common_modules import CommonEncDecModules



def gcd(a, b):
    result = min(a, b)
    while result:
        if a % result == 0 and b % result == 0:
            break
        result -= 1
    return result

class SepChannelsSGMMTool(ToolEngine):
    def __init__(self, chs_input, **kwargs):
        """Initialization of a codec for specific components

        Args:
            chs_input (int): in put channel number. For Y component, the value is 1; for UV components, the value is 2.
        """
        super(SepChannelsSGMMTool, self).__init__(**kwargs)
        self.chs_ls = kwargs.get('chs_ls', 128)
        self.chs_input = chs_input
        self.ccs_id = kwargs.get('ccs_id', 0)
        self.chs_in_supp = kwargs.get('chs_in_supp', 0)
        self.chs_ls_supp = kwargs.get('chs_ls_supp', 0)
        self.skip_depth_step = self.ccs_id != 0     # TODO: move this to config file
        self.Ngmm = 3
        self.__internal_range = (0, 255)
        self.kernel_height = 3
        self.kernel_width = 4
        self.downsample_factor = kwargs.get('downsample_factor', 1)
        
        # for tiling
        num_downsampling_layers = 3 - (1 if self.ccs_id != 0 else 0)
        self.alignment_size = tiling.get_alignment_size(num_downsampling_layers)
        num_upsampling_layers = 3
        self.alignment_size_dec = tiling.get_alignment_size(num_upsampling_layers)

        self.MSprecisionH = 13
        self.MSprecisionL = 7
        self.ls_processing = LSProcessing()
        self.common_modules = CommonEncDecModules(self.chs_ls, self.skip_depth_step, self.alignment_size, self.alignment_size_dec)
        flt_func = lambda x: x.endswith("_prim" if self.ccs_id==0 else "_sec")
        self.encoder = create_dataencoder_instance(chs_ls=self.chs_ls, filter_func=flt_func, tools_set_once=True)
        self.decoder = create_datadecoder_instance(chs_ls=self.chs_ls, filter_func=flt_func, enabled=True, remove_unused_tools=False)
        # qp_map needs self.alignment as an input for size
        self.tile_manager_enc = TileManager(alignment_size=self.alignment_size*4,
                                            latent_downscale_factor_y=self.alignment_size,
                                            use_coding_headers=False)
        self.tile_manager_synthesis = TileManager(  alignment_size=self.alignment_size_dec, 
                                                    latent_downscale_factor_y=self.alignment_size_dec,
                                                    stream_header_part='pic_header',
                                                    enable_flag_name=f'synthesis_tile_enable[{self.ccs_id}]')

        self._params_single_model = CcsSingleModelParams(**kwargs)
        self._params_sep_chan_tool = CcsSepChannelParams()
        self._attrs_proxy_common = AttrsProxy(['skip_cube_thr'], [self.common_modules.skip_mode])
        
    def get_alignment_size(self):
        return self.alignment_size

    def get_processed_img_shape(self, s: List[float] = None):
        if s is None:
            s = self.owner.get_processed_img_shape()
        if self.downsample_factor != 1:
            s = Image.scale_size(s, 1.0 / float(self.downsample_factor))
        return s
    
    def get_ls_shape(self, level: int = 0) -> torch.Size:
        h,w = self.get_processed_img_shape()
        return get_size_on_depth(h,w,4, self.skip_depth_step) if level == 0 else get_size_on_depth(h,w,6, self.skip_depth_step)
            

    @property
    def beta_displacement_log(self) -> float:
        return self.common_modules.quantizer.beta_displacement_log

    @beta_displacement_log.setter
    def beta_displacement_log(self, value: float) -> None:
        self.common_modules.quantizer.set_params(log_k=self.common_modules.log_k)
        self.common_modules.quantizer.beta_displacement_log = value
        
    @property
    def decoder_id(self) -> int:
        cur_model_name = self.decoder.tool
        tools_list = self.decoder.get_tools_list()
        return tools_list.index(cur_model_name) if cur_model_name in tools_list else None

    @decoder_id.setter
    def decoder_id(self, idx: int) -> None:
        """Set decoder ID and update models if it is needed
        """
        if idx is not None:
            tools_list = self.decoder.get_tools_list()
            cur_model_name = self.decoder.tool
            new_model_name = tools_list[idx]
            if cur_model_name != new_model_name:
                self.decoder.tool = new_model_name
                self.decoder._params_loaded()
                        
    def get_sgm_entropy_model(self):
        return self.common_modules.entropy        

    @property
    def internal_data_range(self) -> List[int]:
        """Return range of internal data representation, i.e. [0,1]

        Returns:
            list: range of data presentation
        """
        return self.__internal_range
    
    def get_base_tile_params(self, tile_manager):
        tile_manager.NumSamplesInRegion = self.get_owner_param('NumSamplesInRegion', None)
        # Only tested 1024x1024 region size, remove this check if further tests are supported
        #assert tile_manager.NumSamplesInRegion in [-1, 0, 1024 * 1024]
        
        tile_manager.region_partitioning_flag = self.get_owner_param('region_partitioning_flag')
        if tile_manager.region_partitioning_flag:
            tile_manager.numHorRegions = self.get_owner_param('numHorRegions')
            tile_manager.numVerRegions = self.get_owner_param('numVerRegions')
        else:
            tile_manager.numHorRegions = 1
            tile_manager.numVerRegions = 1
        tile_manager.HyperDecoderOverlap = self.get_owner_param('HyperDecoderOverlap')
        tile_manager.McmOverlap = self.get_owner_param('McmOverlap')
        # 1 indicates using marker-based bitstream structure, and all regions are independent
        tile_manager.region_residual_in_its_own_substream_flag = self.get_owner_param('region_residual_in_its_own_substream_flag')
        
    # TODO: compare this to my other branch. in particual model.alignment_size_dec is not used for decoder setup here
    def cfg_update_for_conformance(self, full_img_height, full_img_width):
        """Function for updating analysis/synthsis transform tile size such that when region_residual_in_its_own_substream_flag is true, multiple integer number of synthesis transform tiles are comprised within a region.
        Args:
            full_img_height: height of the luma component of image in pixel domain.
            full_img_width: width of the luma component of image in pixel domain.
            model: model primary or secondary component processing
        """
        totalRegions = self.common_modules.tile_manager_hyper.numHorRegions * self.common_modules.tile_manager_hyper.numVerRegions
        _, img_width = self.get_processed_img_shape()
        if self.common_modules.tile_manager_hyper.region_partitioning_flag and self.common_modules.tile_manager_hyper.region_residual_in_its_own_substream_flag:
            verRegionSize = (((full_img_height + 127) // 128) // self.common_modules.tile_manager_hyper.numHorRegions) * 128
            horRegionSize = (((full_img_width + 127) // 128) // self.common_modules.tile_manager_hyper.numVerRegions) * 128
            if verRegionSize <= full_img_height and horRegionSize<= full_img_width:
                tile_size = gcd(verRegionSize,horRegionSize)
            elif verRegionSize <= full_img_height:
                tile_size = verRegionSize
            elif horRegionSize<= full_img_width:
                tile_size = horRegionSize
            else:
                tile_size = 10000
            diviser = round(full_img_width/img_width)**2
            self.tile_manager_synthesis.tile_size = tile_size
            self.tile_manager_synthesis.numSamplesPerTile = tile_size**2
            self.tile_manager_enc.numSamplesPerTile = tile_size**2/diviser        
    
    def setup_enc_tile_managers_of_model(self) -> TileManager:
        full_img_height, full_img_width = self.get_original_img_shape()
        img_height, img_width = self.get_processed_img_shape()
        # coeff = model.alignment_size_dec // model.alignment_size
        img_shape = (1,1,img_height, img_width)
        full_img_shape = (1,1,full_img_height, full_img_width)
        # out_img_shape = (1,1,img_height*coeff, img_width*coeff)
        latent_y_height = math.ceil(img_height / self.alignment_size)
        latent_y_width = math.ceil(img_width / self.alignment_size)
        latent_y_shape = (1, self.chs_ls, latent_y_height, latent_y_width)
        latent_psi_height = math.ceil(img_height / (self.alignment_size * 2 ))
        latent_psi_width = math.ceil(img_width / (self.alignment_size * 2 ))
        latent_psi_shape = (1, self.chs_ls*4, latent_psi_height, latent_psi_width)
        latent_z_height = math.ceil(img_height / (self.alignment_size * 4 ))
        latent_z_width = math.ceil(img_width / (self.alignment_size * 4 ))
        latent_z_shape = (1, self.chs_ls*4, latent_z_height, latent_z_width)
        self.get_base_tile_params(self.common_modules.tile_manager_hyper)
        self.cfg_update_for_conformance(full_img_height, full_img_width)
        
        self.get_base_tile_params(self.tile_manager_enc)
        self.get_base_tile_params(self.tile_manager_synthesis)
        self.get_base_tile_params(self.common_modules.tile_manager_hd)
        self.get_base_tile_params(self.common_modules.tile_manager_mcm)

        # model.tile_manager_hd.setup_tiles_enc(full_img_shape, latent_y_shape, latent_psi_shape, latent_z_shape)
        # model.tile_manager_mcm.setup_tiles_enc(full_img_shape, latent_y_shape, latent_psi_shape, latent_z_shape)
        self.common_modules.tile_manager_hyper._init(latent_y_shape, self.alignment_size)
        self.common_modules.tile_manager_hd.setup_tiles_dec_from_region_tile_manager(self.common_modules.tile_manager_hyper,
                                                                       full_img_shape, latent_y_shape, latent_psi_shape,
                                                                       latent_z_shape,
                                                                       alighment_size=self.alignment_size)
        self.common_modules.tile_manager_mcm.setup_tiles_dec_from_region_tile_manager(self.common_modules.tile_manager_hyper,
                                                                        full_img_shape, latent_y_shape,
                                                                        latent_psi_shape, latent_z_shape,
                                                                        alighment_size=self.alignment_size)
        self.tile_manager_enc.setup_tiles_enc(img_shape, latent_y_shape, latent_psi_shape, latent_z_shape)
        self.tile_manager_synthesis.setup_tiles_enc(full_img_shape, latent_y_shape, latent_psi_shape, latent_z_shape)
        
    
    def init_decisions(self) -> Decisions:
        img_height, img_width = self.get_processed_img_shape()

        latent_y_height = math.ceil(img_height / self.alignment_size)
        latent_y_width = math.ceil(img_width / self.alignment_size)
        latent_y_shape = (1, self.chs_ls, latent_y_height, latent_y_width)

        latent_z_height = math.ceil(img_height / (self.alignment_size * 4))
        latent_z_width = math.ceil(img_width / (self.alignment_size * 4))
        latent_z_shape = (1, self.chs_ls, latent_z_height, latent_z_width)

        latent_psi_height = 2* math.ceil(img_height / (self.alignment_size * 4))
        latent_psi_width = 2* math.ceil(img_width / (self.alignment_size * 4))
        latent_psi_shape = (1, self.chs_ls*4, latent_psi_height, latent_psi_width)

        ans = Decisions()
        ans['z_hat'] = torch.zeros(latent_z_shape, device=self.device, dtype=torch.int8)
        ans['psi'] = torch.zeros(latent_psi_shape, device=self.device, dtype=torch.float)
        ans['residual_quant'] = torch.zeros(latent_y_shape, device=self.device, dtype=torch.int16)
        ans['residual'] = torch.zeros(latent_y_shape, device=self.device, dtype=torch.float)
        ans['y'] = torch.zeros_like(ans['residual'])
        ans['scale_log'] = torch.zeros_like(ans['residual'])
        ans['skip_scale_log'] = torch.zeros_like(ans['residual'])
        ans['quantizer'] = Decisions()
        ans['quantizer']['gain_unit'] = Decisions()
        ans['cube_flags'] = torch.zeros_like(ans['residual']).to(dtype=torch.bool)


        ans['psi_tiles'] = dict()
        ans['scale_log_origin_tiles'] = dict()
        ans['scale_log_tiles'] = dict()
        ans['quantizer_tiles'] = dict()
        ans['skip_scale_log_tiles'] = dict()
        ans['cube_flags_tiles'] = dict()
        ans['residual_tiles'] = dict()
        ans['residual_dq_tiles'] = dict()
        ans['y_hat_tiles'] = dict()

        return ans

        
        
    def analysis_and_hyper_encoder(self, img: torch.Tensor, decisions: Decisions):
        # do things for which we only need encoder tile configuration: analysis transform and hyper-analysis transform

        self.logger.debug(
            f'Analysis transforms (y and z) using these tiles on image level: {repr(self.tile_manager_enc.image_tiles)}, Y-hat: {repr(self.tile_manager_enc.latent_tiles)}, Z-hat: {repr(self.tile_manager_enc.latent_tiles_z)}')

        self.logger.debug(f'clipping mode: {self.common_modules.clipping_mode}')
        self.get_profilers().start('CCS Y analysis')
        for tile_info in tiling.ColocatedTiles.iter_colocated_grids(self.tile_manager_enc):
            self.common_modules.compress_colocated_tiles(img, decisions, tile_info)
        self.get_profilers().finish('CCS Y analysis')

    @determinism
    def compress(self,
                  img: torch.Tensor = None,
                  decision: Decisions = {}) -> Decisions:

        """Compress input image to decisions

         Args:
             img (torch.Tensor): input img components: size of Y component is [1,1,height,width], size of UV components is [1,2,height,width].
             decision (Decisions, optional): Defaults to {}.
             sup_info (torch.Tensor, optional): support information. Defaults to None.

         Returns:
             decisions (Decisions): a dict o save the output of the compress function. it contains latent and hyper latent tensor.
        """
        if 'y' not in decision:
            decision = self.init_decisions()
            self.setup_enc_tile_managers_of_model()
            self.analysis_and_hyper_encoder(img, decision)
            

             
        img_height, img_width = img.shape[2:]
        y = decision['y']
        decision = self.common_modules.compress(y, h=img_height, w=img_width, decision=decision)
           

        return decision



    @determinism
    def decompress(self,
                   decisions: Decisions,
                   h: int,
                   w: int,
                   sup_info: torch.Tensor = None) -> torch.Tensor:
        """decompress function will generate reconstructed image

        Args:
            decisions (Decisions): a dict has latent and hyper latent information.
            h (int): height of the image.
            w (int): width of the image.
            sup_info (torch.Tensor, optional): support information. Defaults to None.
        Returns:
            rec (torch.Tensor): reconstructed image.
        """
        # h2,w2 = self.get_processed_img_shape((h, w))
        # self.common_modules.decompress(decisions, h=h2, w=w2)
        
        if 'y_hat' in decisions:
            # with self.get_profilers_ctx(f'{self.name} ls_postprocessing'):
            #     self.ls_processing.post_processing(decisions)      
            y_hat = decisions.get('y_hat')

            with self.get_profilers_ctx('CCS decompress_y'):
                img_height, img_width = h, w
                img_shape = (1, self.chs_input, h, w)
                #latent_height = math.ceil(img_height / self.alignment_size_dec)
                #latent_width = math.ceil(img_width / self.alignment_size_dec)
                #latent_shape = (1, self.chs_ls, latent_height, latent_width)

                if sup_info is not None:
                    ss = sup_info.shape[2:]
                    ys = y_hat.shape[2:]
                    ms_h = min(ss[0], ys[0])
                    ms_w = min(ss[1], ys[1])
                    sup_info_tile = sup_info[:, :, :ms_h, :ms_w] if (ms_h != ss[0] or ms_w != ss[1]) else sup_info
                    y_hat = torch.cat((sup_info_tile, y_hat), dim=1)

                rec = torch.zeros(img_shape, device=y_hat.device)

                #self.tile_manager_synthesis.setup_tiles_dec(img_shape, latent_shape)

                #self.logger.debug(
                #    f'Decompress using these tiles: {repr(self.tile_manager_synthesis.image_tiles)}, in latent space: {repr(self.tile_manager_synthesis.latent_tiles)}')

                #for tile_info in self.tile_manager_synthesis.get_iter_over_tiles(y_hat, rec):
                #    latent_tile_data = tile_info.get_data()
                #    s = tile_info.output_shape()
                #    image_tile_data = self.decoder(latent_tile_data, h=s[0], w=s[1])
                #    tile_info.assign_data(image_tile_data)
                rec = self.decoder(y_hat, h=h, w=w)

            s = rec.shape[-2:]            
            if s[0] != img_height or s[1] != img_width:
                return rec[:, :, :img_height, :img_width] 
            else:
                return rec
        else:
            return None

    def encode(self, ec, decision: Decisions, *args, **kwargs) -> None:
        self.common_modules.encode(ec, decision, *args, **kwargs)
        
    def decode(self, *args, **kwargs) -> Decisions:
        return self.common_modules.decode(*args, **kwargs)

    def encode_header(self, ec: HeaderCoder):
        multi_threading_r = self.num_threads_r>1
        ec.encode(multi_threading_r, bits_count=1, name=f'multi_threading_r[{self.ccs_id}]')
        if multi_threading_r:
            tmp = math.log2(self.num_threads_r)
            assert abs(tmp - int(tmp)) < 1E-5
            log2_num_threads_r_minus1 = int(tmp) - 1
            ec.encode(log2_num_threads_r_minus1, bits_count=2, name=f'log2_num_threads_r_minus1[{self.ccs_id}]')
        ec.encode(self.common_modules.num_chs, bits_count=8, name=f'num_chs[{self.ccs_id}]')
    
    def decode_header(self, ec: HeaderCoder):
        multi_threading_r = ec.decode([1], bits_count=1, name=f'multi_threading_r[{self.ccs_id}]')
        if multi_threading_r:
            log2_num_threads_r_minus1 = ec.decode([1], bits_count=2, name=f'log2_num_threads_r_minus1[{self.ccs_id}]')
            self.num_threads_r = 1 << (log2_num_threads_r_minus1+1)
        else:
            self.num_threads_r = 1
        
        num_chs = ec.decode([1], bits_count=8, name=f'num_chs[{self.ccs_id}]').clamp(0, self.chs_ls).item()
        self.common_modules.num_chs = num_chs
        self.common_modules.num_decode_chs = min(num_chs,self.common_modules.num_decode_chs)
            
        self._params_loaded()

    def _params_loaded(self) -> None:
        EC = self.get_object_by_url('ce.EC')
        setattr(EC, f'num_threads_r_{self.ccs_id}', self.num_threads_r)
        self.decoder_id = self.get_default_decoder_id()
        
        
    def export_models(self, output_dir: str, opset_version: int):
        output_dir = os.path.join(output_dir, self.name)
        os.makedirs(output_dir, exist_ok=True)
        torch.onnx.export(self.encoder, 
                          torch.rand([1, (self.chs_input + self.chs_in_supp*10), 256, 256], device=self.device), 
                          os.path.join(output_dir, "analysis.onnx"),
                          export_params = True, opset_version = opset_version,
                          input_names=['x'], output_names=['y'],
                          dynamic_axes={'x': [2,3], 'y': [2,3]}
                          )
        torch.onnx.export(self.decoder, 
                          torch.rand([1, self.chs_ls + self.chs_ls_supp, 256, 256], device=self.device), 
                          os.path.join(output_dir, "synthesis.onnx"),
                          export_params = True, opset_version = opset_version,
                          input_names=['y'], output_names=['z'],
                          dynamic_axes={'y': [2,3], 'z': [2,3]}
                          )
