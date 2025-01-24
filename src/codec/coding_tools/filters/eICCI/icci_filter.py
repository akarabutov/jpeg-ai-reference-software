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
import time

import torch
import torch.nn as nn
from typing import List
from pytorch_msssim import ms_ssim
from src.codec.common import Image, determinism_on_eval, tiling
from src.codec.entropy_coding import HeaderCoder

from ...tiling import TileManager
from ..base import FilterBase
##
from .icci_models import ThreeStageYUVLite_DWT_V8_444_jointUV
from .model_idxes import ICCIModelIndexes
from .params import IcciParams
import math
from src.codec.common import (TensorOps)




class EfficientICCIFilter(FilterBase):
    """
        Implements the filter bank interface containing multiple ICCI
        filters.
    """
    internal_range = (0., 1.)
    tile_min_size_for_MSSSIM = 176

    def __init__(self, **kwargs):
        super(EfficientICCIFilter, self).__init__(enable_flag_name="icci_enable_flag", **kwargs)
        self.models = nn.ModuleList()
        self._params_icci = IcciParams()
        # for tiling
        num_downsampling_layers = 3
        alignment_size = tiling.get_alignment_size(num_downsampling_layers)
                
        self.tile_manager = TileManager(alignment_size,
                                        signal_tileSignalingType=False,
                                        latent_downscale_factor_y=1)
        self.models_idxes = ICCIModelIndexes(self.tile_manager)
        self.add_sub_tool(f'models_idxes', self.models_idxes)


    def export_models(self, output_dir: str, opset_version: int):
        from src.codec.common import ModulesContext
        from src.codec.components.base_layers import Conv2di
        os.makedirs(output_dir, exist_ok=True)
        for i,(m, m_name) in enumerate(zip(self.models, self.get_ckpt_names())):
            device = next(m.parameters()).device
            d = torch.rand([1, 1, 256, 256], device=device)
            fn = f"model_{i}.onnx"
            #os.path.splitext(m_name)[0] + ".onnx"
            torch.onnx.export(m, 
                          (d,d,d), 
                          os.path.join(output_dir, fn),
                          export_params = True, opset_version = opset_version,
                          input_names=['Y', 'U', 'V'], output_names=['Y_o', 'U_o', 'V_o'],
                          dynamic_axes={'Y': [2,3], 'U': [2,3], 'V': [2,3]}
                          )


    def build_model(self):
        """
        Builds filter bank of ICCI models
        """
        ckpt_names = self.get_ckpt_names()
        for _ in range(len(ckpt_names)):
            self.models.append(
                ThreeStageYUVLite_DWT_V8_444_jointUV(in_nc=self.in_nc,
                                                    out_nc=self.out_nc,
                                                    nf=self.nf,
                                                    nbY=self.nbY,
                                                    nbUV=self.nbUV))
        if self.loss_type == 'mse':
            self.loss = self.calculate_mse
        elif self.loss_type == 'ms-ssim':
            self.loss = self.calculate_msssim
        elif self.loss_type == 'mixed':
            self.loss = self.calculate_mixed

        lw = self.luma_loss_weights
        cw = self.chroma_loss_weights

        self.loss_weights = torch.tensor([lw, cw, cw]).T
        self.skip_luma = sum(self.luma_loss_weights) == 0
        self.skip_msssim = self.loss_weights[1].sum() == 0

        self.model_loss_type = []
        


    def load_model(self, downloader):
        """
        Loads parameters of each model in the filter bank from checkpoint
        files

        Args:
            downloader: Downloader object handling the location of the
                checkpoint files
        """
        cp_model_name = self.get_model_name()
        logger = self.logger

        for i, cp_fn in enumerate(self.get_ckpt_names()):
            full_cp_fn = downloader.get_file_path(cp_model_name, cp_fn)
            if "mse" in cp_fn:
                self.model_loss_type.append('mse')
            else:
                self.model_loss_type.append('msssim')
            if full_cp_fn is not None:
                cp = torch.load(full_cp_fn)
                cp = torch.load(full_cp_fn)
                logger.info(f'Loading checkpoint for {self.get_tool_url()} from a file {full_cp_fn}')
                self.load_state_dict(i, cp)
                self.models[i].to(self.device)


    def load_state_dict(self, model_idx, checkpoint, strict: bool = True):
        """
        Loads parameters of a single model from checkpoint file

        Args:
            model_idx: Index of the model in the filter bank
            checkpoint: File path to the checkpoint file
            strict: whether to strictly enforce that the keys
                in state_dict match the keys returned by this
                module's state_dict() function. Default: True

        Returns:
            Returns a list of str containing the missing keys and
            a list of str containing the unexpected keys
        """
        mk, uek = self.models[model_idx].load_state_dict(checkpoint, strict=strict)
        return mk, uek

    def state_dict(self, model_idx: int, keep_vars=False):
        """
        Returns a dictionary containing a whole state of a filter with
        given model_idx.

        Args:
            model_idx: Index of the model in the filter bank

        Returns:
            State dictionary containing a whole state of a filter
        """
        return self.models[model_idx].state_dict(keep_vars=keep_vars)

    def _get_models_location(self, ):
        """
        Returns the device on which the model parameters are allocated.

        Returns:
            Returns a device type ('cpu' or 'cuda')
        """
        return self.device

    def compress_tile(self, tile_data: torch.Tensor, tile_dwt: torch.Tensor, model_idx:int, comp_idx:int) -> Image:
        """
        Applies the filter with the given model index to a single image tile
        and returns the filtered image tile. This operation is performed
        during the encoding process.

        Args:
            tile_data: ``torch.Tensor`` containing the image tile
            model_idx: Index of the model in the filter bank

        Returns:
            Returns the filtered image tile as ``Image`` object
        """
        with torch.no_grad():

            if model_idx != -1 and ((comp_idx==0 and not self.skip_luma) or comp_idx==1):
                ans = self.active_models[model_idx].process_comp(*tile_dwt, comp_idx=comp_idx)
            else:
                ans = tile_data[:,[0]]

        return ans


    def decompress_tile(self, tile_data: torch.Tensor, tile_idx: int) -> torch.Tensor:
        """
        Apply the filter selected during the encoding process to a single
        image tile and returns the filtered image tile. This operation is
        performed during the decoding process.

        Args:
            tile_data: ``torch.Tensor`` containing the image tile
            model_idx: Index of the model in the filter bank

        Returns:
            Returns the filtered image tile as ``Image`` object
        """

        ans = tile_data.clone()

        tile_data = tile_data.to(dtype=torch.float)
        tile_shape = tile_data.shape
        tile_data = padding_layer(tile_data)

        model_selection = self.models_idxes.get_model_index(tile_idx)

        with torch.no_grad():
            if any(model_selection):
                modelY, modelU, modelV = model_selection
                modelUV = modelU or modelV

                tmp = self.active_models[0].process_dwt444_2(tile_data)
                if modelY:
                    out = self.active_models[modelY - 1].process_comp(*tmp, comp_idx=0)[:, :, :tile_shape[-2], :tile_shape[-1]]
                    ans[:, [0]] = out
                    
                if modelUV:
                    out = self.active_models[modelUV - 1].process_comp(*tmp, comp_idx=1)[:, :, :tile_shape[-2], :tile_shape[-1]]
                    if modelU:
                        ans[:, [1]] = out[:, [0]]
                    if modelV:
                        ans[:, [2]] = out[:, [1]]

        return ans

    @staticmethod
    def log_formater(x):
        x = x.tolist() if x.shape else [x.item()]
        return ','.join([f'{i:.5f}' for i in x])

    def get_current_error(self, tile_org, tile_rec):
        cur_err = self.loss(tile_org, tile_rec)
        y_error = cur_err[0]
        uv_error = torch.cat([torch.sum(cur_err[1:]).reshape(1),
                              cur_err[1:]])
        return y_error, uv_error

    def get_base_model_info(self):
        base_model_id = self.get_base_model_id()
        decoder_id = self.get_default_decoder_id()
        base_model_type = 'sop' if decoder_id == 0 else 'bop' if decoder_id == 1 else 'hop'
        return base_model_id, base_model_type
    
    def initialize_models_idxes(self):
        base_model_id, base_model_type = self.get_base_model_info()
        eicci_type = 'bop' if base_model_type =='sop' else base_model_type 
        self.models_idxes.base_model_type = base_model_type # bop sop
        self.models_idxes.base_model_id = base_model_id # 0==006, 1==012 ....
        self.active_models = [m for (m, ckpt) in zip(self.models, self.get_ckpt_names()) if eicci_type in ckpt]

        all_models = {str(i):list(range(len(self.active_models))) for i in range(5)}
        model_idx_map_long = {"y": all_models,
                            "uv": all_models}
        model_idx_map_short = {"y": self.y_short_list,
                            "uv": self.uv_short_list}

        self.models_idxes.model_idx_map_long = model_idx_map_long
        self.models_idxes.model_idx_map_short = model_idx_map_short
        self.models_idxes.use_short_list = self.process_short_list # this will be overwritten during the decoding 

    def auto_enableflag_detected_value(self) -> bool:
        if self.get_owner_param('s_ver')!=1 and self.get_owner_param('s_hor')!=1:
            return False
        return None


    @determinism_on_eval
    def compress(self, imgs: List[Image], org_img_i: Image, *args, **kwargs) -> List[Image]:
        """
        This is the encoder side implementation of the filter process.
        It applies the whole filter bank a single (compressed) image during
        the encoding process. A brute force approach is used to find the
        best filter combination per image channel according to the
        selected loss function ``self.loss_type`` ('mse' or 'ms-ssim').
        Loss function is a reference based metric, therefore, original
        image (uncompressed) also required for the processing. If the
        performance of the filtered image for an image channel is lower
        than the compressed image, then no filter will be applied for
        that channel. The filter selection is stored in
        ``self.models_idxes`` which later will be encoded into the
        bitstream.

        If the resolution of the compressed image is larger
        than the available memory the images will be processed in tiles
        and filter selection will be stored per image channel of each
        tile.

        Args:
            img: Compressed image reconstructed by the decoder
            org_img_i: Uncompressed image used as reference for the
                filter selection

        Returns:
            Returns the filtered image
        """

        if self.get_owner_param('s_ver')!=1 or self.get_owner_param('s_hor')!=1:
            self.set_enable(False)
            return imgs
        
        self.initialize_models_idxes()

        img = imgs[0]

        
        test_s_time = time.time()
        _, _, h, w = img.shape
        input_range = img.data_range
        img.convert_range_(self.internal_range)
        org_img_i.convert_range_(self.internal_range)

        img.to_YUV_()
        org_img_i.to_YUV_()

        img_rec = img.clone()
        img_rec.to_444_()
        img_rec = img_rec.get_tensor()

        org_img = org_img_i.clone()
        org_img.to_444_()
        org_img = org_img.get_tensor()

        img_flt = img_rec.clone()

        self.tile_manager.setup_tiles_enc(img.shape,
                                          None,
                                          None,
                                          None,
                                          minimum_tile_size=self.tile_min_size_for_MSSSIM)

        img_tmp = torch.cat([org_img, img_rec], dim=0)
        logger = self.logger
        log_formatter = self.log_formater

        logger.debug(f'eICCI using these tiles:\n {repr(self.tile_manager.image_tiles)}')
        if self.loss_type == 'mixed':
            logger.debug(f'Mixed loss coeffs:\n {self.loss_weights}')

        tile_idx = 0
        for tile_info in self.tile_manager.image_tiles:

            tile_org = tiling.get_data(org_img, tile_info)
            tile_rec = tiling.get_data(img_rec, tile_info)
            tile_enhanced = tile_rec.clone()

            running_y_error, running_uv_error = self.get_current_error(tile_org,
                                                                       tile_rec)

            initial_y_error = running_y_error.clone()
            initial_uv_error = running_uv_error.clone()

            uv_gain = initial_uv_error - running_uv_error
            gain_index = torch.argmax(uv_gain)

            current_selection = [0] * len(Image.valid_comp_names)
            
            # process with all models
            current_tile_enhanced = tile_rec.clone()

            
            tile_rec = tile_rec.to(dtype=torch.float)
            tile_shape = tile_rec.shape
            tile_rec_padded = padding_layer(tile_rec)
            tile_dwt = self.active_models[0].process_dwt444_2(tile_rec_padded)

            # for model_idx in NEWDICT_Y:
            #     RUN ONLY Y AND RECORD ONLY Y

            # for model_idx in NEWDICT_UV:
            #     RUN ONLY UV AND RECORD ONLY UV


            model_indices_y, model_indices_uv = self.models_idxes.map_idx
            # dict = {
            #     0 :[0,4],# here first 0:is the codec idsuch as 006 (QP), in the values 0 means the first filter
            #     1 :[2,7,-1],
            # }
            for model_idx_y, model_idx_uv in zip(model_indices_y, model_indices_uv):
                ans_y = self.compress_tile(tile_rec_padded,
                                           tile_dwt,
                                           model_idx_y,
                                           0)
                ans_uv = self.compress_tile(tile_rec_padded,
                            tile_dwt,
                            model_idx_uv,
                            1)

                current_tile_enhanced = torch.cat([ans_y, ans_uv], dim=1)
                current_tile_enhanced = current_tile_enhanced[:, :, :tile_shape[-2], :tile_shape[-1]]
                current_tile_enhanced = torch.clamp(current_tile_enhanced, *self.internal_range).to(self.device)

                current_y_error, current_uv_error = self.get_current_error(tile_org, current_tile_enhanced)

                # Y loss
                if current_y_error < running_y_error:
                    tile_enhanced[:, [0]] = current_tile_enhanced[:, [0]]
                    running_y_error = current_y_error
                    current_selection[0] = model_idx_y + 1


                cur_uv_gain = initial_uv_error - current_uv_error
                cur_gain_index = torch.argmax(cur_uv_gain)

                # UV loss
                if model_idx_uv != -1 and cur_uv_gain[cur_gain_index] > uv_gain[gain_index]:
                    running_uv_error = current_uv_error
                    uv_gain = cur_uv_gain
                    gain_index = cur_gain_index

                    current_selection[1] = model_idx_uv + 1
                    current_selection[2] = model_idx_uv + 1

                    if gain_index == 0:
                        tile_enhanced[:, [1]] = current_tile_enhanced[:, [1]]
                        tile_enhanced[:, [2]] = current_tile_enhanced[:, [2]]
                    elif gain_index == 1:
                        tile_enhanced[:, [1]] = current_tile_enhanced[:, [1]]
                        tile_enhanced[:, [2]] = tile_rec[:, [2]]
                        current_selection[2] = 0
                    else:
                        tile_enhanced[:, [1]] = tile_rec[:, [1]]
                        tile_enhanced[:, [2]] = current_tile_enhanced[:, [2]]
                        current_selection[1] = 0


                model_idx = [model_idx_y+1, model_idx_uv+1]
                logger.debug(
                    f'Tile: {tile_idx}  Loss: {self.loss_type}  Model: {model_idx}  Current Selection: {current_selection} '
                    f'Initial Loss Y: {log_formatter(initial_y_error)} | '
                    f'Initial Loss UV: {log_formatter(initial_uv_error[1:])} | '
                    f'Current Loss Y: {log_formatter(current_y_error)} | '
                    f'Current Loss UV: {log_formatter(current_uv_error[1:])} | '
                    f'Current Gain UV: {log_formatter(cur_uv_gain)} | ')

            # assign best combination of filters
            assigned_tile, assigned_tile_rel_to_overlap = self.tile_manager.get_core_of_overlapping_image_tile(
                tile_info)
            assigned_tile_data = tiling.get_data(tile_enhanced, assigned_tile_rel_to_overlap)

            tiling.assign_data(img_flt, assigned_tile, assigned_tile_data)

            # store model selection
            self.models_idxes.model_state_per_tile.append(current_selection)

            tile_idx += 1

        test_e_time = time.time()
        logger.info(
            f'eICCI YUV Filter processed (enc) {test_e_time - test_s_time:.5} seconds. Filters {self.models_idxes.model_state_per_tile} used'
        )
        img_flt = Image.create_from_tensor(img_flt, 
                                           self.internal_range, 
                                           bit_depth=img.bit_depth,
                                           color_space='yuv')

        img_flt.convert_range_(input_range)
        output_fmt = Image.get_format_from_subsampling(self.get_owner_param('s_ver'), self.get_owner_param('s_hor'))
        img_flt.to_format_(output_fmt)
        
        # self.logger.debug(f"The eICCI control point is {TensorOps.get_hash(img_flt.get_tensor())}\n")
        return [img_flt] + imgs[1:]

    @determinism_on_eval
    def decompress(self, imgs: List[Image], *args, **kwargs) -> List[Image]:
        """
        This is the decoder side implementation of the filter process.
        It applies a particular filter(s) from the filter bank to the whole
        image according to the filter selection (see ``self.compress``).
        It uses the ``self.models_idxes`` which stores the filter selection
        per channel (and per tile if required). Since this process does
        not use any search on model selection, it performs relatively
        faster than ``self.compress``

        Similar to the encoder side, if the resolution of the compressed
        image is larger than the available memory the images will be
        processed in tiles.

        Args:
            img: Compressed image reconstructed by the decoder

        Returns:
            Returns the filtered image
        """
        
        img = imgs[0]
        test_s_time = time.time()
        input_range = img.data_range
        input_fmt = img.format

        img.to_YUV_()
        img.to_444_()
        img.convert_range_(self.internal_range)

        img_flt = img.get_tensor()
        logger = self.logger
        

        for c in Image.valid_comp_names:
            logger.debug(
                f'Input img info for {c}. Min {img.get_component(c).min()}, max {img.get_component(c).max()}, L1 {img.get_component(c).abs().sum()}'
            )

        tile_idx = 0
        for tile_info in self.tile_manager.image_tiles:
            # tile_org = tiling.get_data(org_img, tile_info)
            rec_data = tiling.get_data(img_flt, tile_info)

            flt_data = rec_data.clone()
            flt_data = self.decompress_tile(rec_data, tile_idx)

            # assign best combination of filters
            assigned_tile, assigned_tile_rel_to_overlap = self.tile_manager.get_core_of_overlapping_image_tile(
                tile_info)
            assigned_tile_data = tiling.get_data(flt_data, assigned_tile_rel_to_overlap)

            tiling.assign_data(img_flt, assigned_tile, assigned_tile_data)

        img_flt = Image.create_from_tensor(img_flt, 
                                           self.internal_range, 
                                           bit_depth=img.bit_depth,
                                           color_space='yuv')

        img_flt.clip_data_()
        test_e_time = time.time()
        logger.debug(
            f'eICCI YUV Filter processed (dec) {test_e_time - test_s_time:.5} seconds. Filters {self.models_idxes.model_state_per_tile} used'
        )
        img_flt.convert_range_(input_range)
        output_fmt = Image.get_format_from_subsampling(self.get_owner_param('s_ver'), self.get_owner_param('s_hor'))
        img_flt.to_format_(output_fmt)
        # self.logger.debug(f"The eICCI control point is {TensorOps.get_hash(img_flt.get_tensor())}\n")
        return [img_flt] + imgs[1:]

    def decode_header(self, ec: HeaderCoder):
        """
        Initializes the tile manager for the decoder side implementation.
        """
        img_height, img_width = self.get_original_img_shape()
        img_shape = (1, 1, img_height, img_width)
        self.tile_manager.init_decoder = lambda: self.tile_manager.setup_tiles_dec(
            img_shape, None, None, None, minimum_tile_size=self.tile_min_size_for_MSSSIM)
        self.initialize_models_idxes()

    @staticmethod
    def calculate_mse(im1: torch.Tensor, im2: torch.Tensor) -> torch.Tensor:
        """
        Returns the Mean Squared Error between im1 and im2

        Args:
            im1: ``torch.Tensor``
            im2: ``torch.Tensor``

        Returns:
            output: ``torch.Tensor``
        """
        return torch.mean((im1 - im2)**2, dim=(0, 2, 3)).cpu()

    @staticmethod
    def calculate_msssim(im1: torch.Tensor, im2: torch.Tensor) -> torch.Tensor:
        """
        Returns the MS-SSIM per image channel between im1 and im2

        Args:
            im1: ``torch.Tensor``
            im2: ``torch.Tensor``

        Returns:
            output: ``torch.Tensor``
        """

        err = []
        for i in range(im1.shape[1]):
            err.append(1.0 - ms_ssim(im1[:, [i]], im2[:, [i]], data_range=1.))
        err = torch.stack(err).cpu()
        return err

    def calculate_mixed(self, im1: torch.Tensor, im2: torch.Tensor) -> torch.Tensor:
        weight = self.loss_weights
        mse = self.calculate_mse(im1, im2)
        if not self.skip_msssim:
            msssim = self.calculate_msssim(im1, im2)
        else:
            msssim = torch.tensor([1,1,1])
        err = torch.stack([mse,msssim])
        err = (weight * err).sum(dim=0)
        return err


def padding_layer(x):
    _,c,h,w = x.shape
    h_diff = int(math.ceil(h / 4) * 4) - h
    w_diff = int(math.ceil(w / 4) * 4) - w
    x = torch.nn.functional.pad(x, (0, w_diff, 0, h_diff), mode='replicate')
    return x