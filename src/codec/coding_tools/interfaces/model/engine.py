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
from ..base import BaseEngine
from .interface import ModelInterface, ModelMgmt
from .params import ModelEngineParams
from src.codec.utils import Downloader


class ModelEngine(ModelInterface, ModelMgmt, BaseEngine):
    def __init__(self, *args, **kwargs):
        super(ModelEngine, self).__init__(*args, **kwargs)
        self._params_model_engine = ModelEngineParams()

    def build_models_recursively(self) -> None:
        self.build_model()
        if hasattr(self, 'device'):
            self.to(self.device)
        self.for_top_level_children(lambda _, m: m.build_models_recursively(), ModelEngine)

    def download_models_recursively(self, downloader: Downloader) -> None:
        self.download_model(downloader)
        self.for_top_level_children(lambda _, m: m.download_models_recursively(downloader),
                                    ModelEngine)

    def load_models_recursively(self, downloader: Downloader) -> None:
        self.load_model(downloader)
        self.for_top_level_children(lambda _n, m: m.load_models_recursively(downloader),
                                    ModelEngine)

        if hasattr(self, 'resume') and self.resume is not None:
            if len(self.resume) == 1:
                self.load_checkpoint(self.resume[0])
            elif len(self.resume) > 1:
                msg = 'Cannot select checkpoint for loading. Bpp is {} for element {}. List of checkpoints: {}'
                msg = msg.format(self.get_target_bpp_idx(), self.get_tool_url(), self.resume)
                self.logger.critical(msg)
            else:
                pass
        self.model_loaded()

    def update_decoder_id_recursively(self, new_decoder_id: int, downloader: Downloader) -> None:
        self.update_decoder_id(new_decoder_id, downloader)
        self.for_top_level_children(lambda _, m: m.update_decoder_id_recursively(new_decoder_id, downloader), ModelEngine)


