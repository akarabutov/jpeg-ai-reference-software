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

import time

import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim
from typing import List
from src.codec.common import Image, determinism_on_eval, tiling
from src.codec.components.base_layers import padding_layer
from src.codec.entropy_coding import HeaderCoder

from ...tiling import TileManager
from ..base import FilterBase
##
from .icci_models import ThreeStageYUVLite_DWT_V3_444
from .model_idxes import ICCIModelIndexes
from .params import IcciParams


class ICCIFilter(FilterBase):
    """
        Implements the filter bank interface containing multiple ICCI
        filters.
    """
    internal_range = (0., 1.)
    tile_min_size_for_MSSSIM = 176

    def __init__(self, **kwargs):
        super(ICCIFilter, self).__init__(**kwargs)
        self.models = nn.ModuleList()
        self._params_icci = IcciParams()
        # for tiling
        num_downsampling_layers = 3
        alignment_size = tiling.get_alignment_size(num_downsampling_layers)
        self.tile_manager = TileManager(alignment_size,
                                        latent_downscale_factor_y=1,
                                        numSamplesPerTile=4194304,
                                        numSamplesTileOverlap=48)
        self.models_idxes = list()

        for c in range(len(Image.valid_comp_names)):
            tool = ICCIModelIndexes(self.tile_manager)
            self.models_idxes.append(tool)
            self.add_sub_tool(f'models_idxes_{c}', tool)

    def build_model(self):
        """
        Builds filter bank of ICCI models
        """
        for _ in range(len(self.get_ckpt_names())):
            self.models.append(
                ThreeStageYUVLite_DWT_V3_444(in_nc=1, out_nc=1, nf=48, nbY=8, nbUV=13))
        if self.loss_type == 'mse':
            self.loss = self.calculate_mse
        elif self.loss_type == 'ms-ssim':
            self.loss = self.calculate_msssim
        elif self.loss_type == 'mixed':
            self.loss = self.calculate_mixed

        lw = self.luma_loss_weights
        cw = self.chroma_loss_weights

        self.loss_weights = torch.tensor([lw, cw, cw]).T

        for model in self.models_idxes:
            model.models_count = len(self.get_ckpt_names())

    def load_model(self, downloader):
        """
        Loads parameters of each model in the filter bank from checkpoint
        files

        Args:
            downloader: Downloader object handling the location of the
                checkpoint files
        """
        cp_model_name = self.get_model_name()

        for i, cp_fn in enumerate(self.get_ckpt_names()):
            full_cp_fn = downloader.get_file_path(cp_model_name, cp_fn)
            cp = torch.load(full_cp_fn)
            self.logger.info(f"=> loading checkpoint '{cp_fn}'")
            self.load_state_dict(i, cp)
            self.logger.info(f'=> loaded checkpoint {cp_fn}')
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

    def compress_tile(self, tile_data: torch.Tensor, model_idx) -> Image:
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
        tile_data = tile_data.to(dtype=torch.float)
        tile_shape = tile_data.shape
        tile_data = padding_layer(tile_data, tile_shape[-2], tile_shape[-1], 0)

        with torch.no_grad():
            ans = self.models[model_idx](tile_data[:, [0]], tile_data[:, [1]], tile_data[:, [2]])
            ans = torch.cat(ans, dim=1)
            ans = torch.clamp(ans[:, :, :tile_shape[-2], :tile_shape[-1]],
                              *self.internal_range).to(self.device)
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
        tile_data = padding_layer(tile_data, tile_shape[-2], tile_shape[-1], 0)

        process_component = {}  # stores idx of the model per component
        for i, m in enumerate(self.models_idxes):
            model_idx = m.get_model_index(tile_idx)
            if model_idx not in process_component:
                process_component[model_idx] = []
            process_component[model_idx].append(i)

        with torch.no_grad():
            for model_idx, comp_idx in process_component.items():
                if model_idx > 0:
                    tmp = self.models[model_idx - 1].process_dwt444(tile_data)
                    for c in comp_idx:
                        ans[:, [c]] = self.models[model_idx - 1].process_comp(
                            *tmp, comp_idx=c)[:, :, :tile_shape[-2], :tile_shape[-1]]
        return ans

    @staticmethod
    def log_formater(x):
        return ','.join([f'{i:.5f}' for i in x.tolist()])

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
        img = imgs[0]
        test_s_time = time.time()
        _, _, h, w = img.shape
        input_range = img.data_range
        REC_IMG = img.clone()
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
                                          img.shape,
                                          img.shape,
                                          minimum_tile_size=self.tile_min_size_for_MSSSIM)

        img_tmp = torch.cat([org_img, img_rec], dim=0)
        logger = self.logger
        log_formatter = self.log_formater

        logger.debug(f'ICCI using these tiles:\n {repr(self.tile_manager.image_tiles)}')
        if self.loss_type == 'mixed':
            logger.debug(f'Mixed loss coeffs:\n {self.loss_weights}')

        for tile_info in self.tile_manager.get_iter_over_tiles(img_tmp, img_flt):
            tmp = tile_info.get_data()
            tile_org, tile_rec = torch.chunk(tmp, 2, dim=0)
            tile_enhanced = tile_rec.clone()

            running_err = self.loss(tile_org, tile_rec)
            initial_error = running_err.clone()  # just for debugging purposes
            current_selection = [0] * len(Image.valid_comp_names)
            # process with all models
            for model_idx in range(len(self.models)):
                current_tile_enhanced = self.compress_tile(tile_rec, model_idx)
                current_err = self.loss(tile_org, current_tile_enhanced)
                # logger.debug(f"model_idx {model_idx} {self.loss_type} Loss: {current_err}")

                # calculate loss, store loss and processed tile temporary
                for i, c in enumerate(Image.valid_comp_names):
                    if current_err[i] < running_err[i]:
                        tile_enhanced[:, [i]] = current_tile_enhanced[:, [i]]
                        running_err[i] = current_err[i]
                        current_selection[i] = model_idx + 1
                logger.debug(
                    f'Tile: {tile_info.get_idx()}  Loss: {self.loss_type}  Model: {model_idx+1}  Current Selection: {current_selection} '
                    f'Loss:  Initial {log_formatter(initial_error)} | '
                    f'Current: {log_formatter(running_err)} | '
                    f'Model[{model_idx+1}]: {log_formatter(current_err)}')

            # assign best combination of filters
            tile_info.assign_data(tile_enhanced)
            # store model selection
            for i, m in enumerate(current_selection):
                self.models_idxes[i].model_state_per_tile.append(m)

        test_e_time = time.time()
        logger.info(
            f'Torch ICCI YUV Filter processed {test_e_time - test_s_time:.5} seconds. Filters {[i.model_state_per_tile for i in self.models_idxes]} used'
        )
        img_flt = Image.create_from_tensor(img_flt, self.internal_range, color_space='yuv')
        img_flt.convert_range_(input_range)

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

        img.to_YUV_()
        img.to_444_()
        img.convert_range_(self.internal_range)

        im_flt = img.clone()
        logger = self.logger

        for c in Image.valid_comp_names:
            logger.debug(
                f'Input img info for {c}. Min {img.get_component(c).min()}, max {img.get_component(c).max()}, L1 {img.get_component(c).abs().sum()}'
            )

        for tile_info in self.tile_manager.get_iter_over_tiles(img.get_tensor(), im_flt):
            rec_data = tile_info.get_data()
            flt_data = self.decompress_tile(rec_data, tile_info.get_idx())
            tile_info.assign_data(flt_data)

        im_flt.clip_data_()
        test_e_time = time.time()
        logger.debug(
            f'Torch ICCI YUV Filter processed {test_e_time - test_s_time:.5} seconds. Filters {[x.get_model_indexes() for x in self.models_idxes]} used'
        )
        im_flt.convert_range_(input_range)
        return [im_flt] + imgs[1:]

    def decode_header(self, ec: HeaderCoder):
        """
        Initializes the tile manager for the decoder side implementation.
        """
        img_height, img_width = self.get_original_img_shape()
        img_shape = (1, 1, img_height, img_width)
        self.tile_manager.init_decoder = lambda: self.tile_manager.setup_tiles_dec(
            img_shape, img_shape, minimum_tile_size=self.tile_min_size_for_MSSSIM)

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
        err = []
        weight = self.loss_weights

        err.append(self.calculate_mse(im1, im2))
        err.append(self.calculate_msssim(im1, im2))
        err = torch.stack(err)

        err = (weight * err).sum(dim=0) / weight.sum(dim=0)
        return err
