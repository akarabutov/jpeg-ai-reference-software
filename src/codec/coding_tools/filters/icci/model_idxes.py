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

from typing import List, Union

from src.codec.entropy_coding import HeaderCoder

##
from ...interfaces import CoderEngine
from ...tiling import TileManager


class ICCIModelIndexes(CoderEngine):
    """
        Implements a database containing selected filters from a filter
        bank. On the encoder side, this class helps storing filter
        selection during the encoding process and encoding them into the
        bitstream. On the decoder side, it reads the filter selection from
        the bitstream and stores them for the filter application.
    """
    def __init__(self, tile_manager: TileManager, *args, **kwargs):
        super(ICCIModelIndexes, self).__init__(has_enabled_flag=False, *args, **kwargs)
        self.__models_count = 0        
        self.__dict__['tile_manager'] = tile_manager

    @property
    def models_count(self) -> int:
        return self.__models_count

    @models_count.setter
    def models_count(self, value: int) -> None:
        self.__models_count = value
        
    def _params_loaded(self) -> None:
        self.model_state_per_tile = list()
        self.model_state = None
        self.use_single_model = False
        self.use_default_model = False

    def get_model_indexes(self) -> Union[int, List]:
        """
        Returns selected model indexes per tile
        """

        return self.model_state_per_tile

    def get_model_index(self, tile_idx: int) -> int:
        """
        Returns selected model indexes for a given tile index
        """

        return self.model_state_per_tile[tile_idx]

    def decode_header(self, ec: HeaderCoder):
        """
        Decodes model selection from the bitstream using the Entropy
        Coder Engine.

        Args:
            ec: `Entropy Coder Engine
        """

        tile_manager = self.__dict__['tile_manager']
        logger = self.logger

        if tile_manager.is_enabled():
            self.use_single_model = bool(ec.decode([1], 1, name='icci_use_single_model'))
            if self.use_single_model:
                model_idx = int(ec.decode([1], self.__models_count, name='icci_model_idx'))
                for idx, tile in enumerate(tile_manager.image_tiles):
                    self.model_state_per_tile.append(model_idx)
                    logger.debug(f'PPS for tile {tile}: {self.model_state_per_tile[idx]}')
            else:
                self.use_default_model = bool(ec.decode([1], 1, name='icci_use_default_model'))
                if self.use_default_model:
                    default_model_idx = int(
                        ec.decode([1], self.__models_count, name='icci_default_model_idx'))
                    for idx, tile in enumerate(tile_manager.image_tiles):
                        use_default_idx = bool(ec.decode([1], 1, name='icci_use_default_idx'))
                        if use_default_idx:
                            self.model_state_per_tile.append(default_model_idx)
                        else:
                            model_idx = int(
                                ec.decode([1], self.__models_count, name='icci_model_idx'))
                            self.model_state_per_tile.append(model_idx)
                            logger.debug(f'PPS for tile {tile}: {self.model_state_per_tile[idx]}')
                else:
                    for idx, tile in enumerate(tile_manager.image_tiles):
                        self.model_state_per_tile.append(
                            int(ec.decode([1], self.__models_count, name='icci_model_idx')))
                        logger.debug(f'PPS for tile {tile}: {self.model_state_per_tile[idx]}')
        else:
            model_idx = int(ec.decode([1], self.__models_count, name='icci_model_idx'))
            self.model_state_per_tile.append(model_idx)
            logger.debug(f'PPS: {self.model_state_per_tile[0]}')

    def encode_header(self, ec: HeaderCoder):
        """
        Encodes model selection into the bitstream using the Entropy
        Coder Engine.

        Args:
            ec: Entropy Coder Engine
        """

        tile_manager = self.__dict__['tile_manager']
        logger = self.logger

        if tile_manager.is_enabled():
            ec.encode(self.use_single_model, 1, name='use_single_model')
            if self.use_single_model:
                raise NotImplementedError
            else:
                ec.encode(self.use_default_model, 1, name='use_default_model')
                if self.use_default_model:
                    raise NotImplementedError
                else:
                    for idx, tile in enumerate(tile_manager.image_tiles):
                        logger.debug(f'PPS and tile {tile}: {self.model_state_per_tile[idx]}')
                        ec.encode(self.model_state_per_tile[idx],
                                          self.__models_count,
                                          name='model_idx')
        else:
            logger.debug(f'PPS: {self.model_state_per_tile[0]}')
            ec.encode(self.model_state_per_tile[0], self.__models_count, name='model_idx')
