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

from typing import List

import torch

from src.codec.common import Decisions, Image
from src.codec.utils import Downloader

##
from .params import ModelMgmtParams


class ModelInterface(object):
    def __init__(self, *args, **kwargs):
        super(ModelInterface, self).__init__(*args, **kwargs)

    # Virtual function
    def build_model(self):
        """
        Build your model here. All parameters are already loaded
        """
        pass
    
    def model_loaded(self):
        """
        Model loaded
        """
        pass

    # The first stage on encoder-side: compression to latent representation
    def compress(self, image: Image, *args, **kwargs) -> Decisions:
        """
        Compress the current image and return encoder decisions
        """
        raise NotImplementedError

    # The second stage on decoder-side: decompression latent representation to image
    def decompress(self, decisions: Decisions, ref_img=None, *args, **kwargs) -> Image:
        """
        Restore the current image based on decision and return reconstructed image
        """
        raise NotImplementedError


class ModelMgmt(object):
    # Public functions
    def __init__(self, *args, **kwargs):
        super(ModelMgmt, self).__init__(*args, **kwargs)
        self.model_params = ModelMgmtParams()
        self.default_model_name = str()
        self.default_ckpt_names = list()

    def hasattr_model_name(self):
        return hasattr(self, 'ckpt_model_name') and self.ckpt_model_name is not None

    def hasattr_ckpt_names(self):
        return hasattr(self, 'ckpt_files') and len(self.ckpt_files) > 0

    def get_model_name(self) -> str:
        """Get name of model (see models.json)

        Returns:
            str: name of selected model
        """
        ans = self.default_model_name
        if self.hasattr_model_name():
            ans = self.ckpt_model_name
        return ans

    def get_ckpt_names(self) -> List[str]:
        """List of checkpoints in selected model

        Returns:
            List[str]: list of files with checkpoint
        """
        ans = self.default_ckpt_names
        if self.hasattr_ckpt_names():
            ans = self.ckpt_files
        return ans

    def download_model(self, downloader: Downloader):
        cp_model_name = self.get_model_name()
        if len(cp_model_name) > 0:
            self.logger.info(f'Downloading model {cp_model_name}')
            downloader.download_models([cp_model_name])

    def load_model(self, downloader: Downloader):
        cp_model_name = self.get_model_name()
        cp_files = self.get_ckpt_names()

        if len(cp_model_name) > 0 and len(cp_files) > 0:
            cp_fn = ''
            if len(cp_files) == 1:
                cp_fn = cp_files[0]
            else:
                cp_fn = self._select_checkpoint(cp_files)
                if (cp_fn is None) or (cp_fn not in cp_files):
                    raise ValueError

            full_cp_fn = downloader.get_file_path(cp_model_name, cp_fn)
            if full_cp_fn is not None:
                self.load_checkpoint(full_cp_fn, strict=True, map_location=self.device)
                self.clipping_mode = self.clipping_mode_opt

    def load_checkpoint(self, filename, map_location=None, strict=False):
        logging = self.logger
        logging.info(f'Loading checkpoint for {self.get_tool_url()} from a file {filename}')
        cp = torch.load(filename, map_location=map_location)
        keys2remove = ['best_loss', 'optimizer', 'channel_wise_entropy']
        for k in keys2remove:
            if k in cp:
                del cp[k]
        mk, uek = self.load_state_dict(cp, strict=strict)
        if len(mk) > 0:
            logging.warn(
                f'Missed {len(mk)} keys during loading checkpoint for {self.get_tool_url()}')
        if len(uek) > 0:
            logging.error(
                f'Got {len(uek)} unexpected keys during loading checkpoint for {self.get_tool_url()}'
            )

    # Virtual functions
    def _select_checkpoint(self, cp_fns: List[str]) -> str:
        """Select final checkpoint if several were presented

        Args:
            cp_fns (List[str]): list of checkpoints

        Returns:
            str: name of checkpoint to be used
        """
        idx = self.get_target_bpp_idx()
        return cp_fns[idx] if (idx < len(cp_fns) and idx >= 0) else ''
    
    def update_decoder_id(new_decoder_id: int, downloader: Downloader) -> None:
        pass
