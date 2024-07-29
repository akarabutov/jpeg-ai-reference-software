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
        self.__dict__['tile_manager'] = tile_manager
        self.__use_short_list = False
        self.__base_model_id = 0
        self.base_model_type = None
        self.model_idx_map_long = {}
        self.model_idx_map_short = {}


    @property
    def base_model_id(self) -> int:
        return self.__base_model_id
    
    @base_model_id.setter
    def base_model_id(self, base_model_id: int) -> None:
        self.__base_model_id = str(base_model_id)

    @property
    def models_count(self) -> int:
        return len(self.map_idx[0])

    @property
    def models_count_long(self) -> int:
        return len(self.map_idx_long[0])

    @property
    def use_short_list(self) -> bool:
        return self.__use_short_list

    @use_short_list.setter
    def use_short_list(self, use_short_list: bool) -> None:
        self.__use_short_list = use_short_list
    
    @property
    def map_idx(self) -> int:
        if self.use_short_list:
            return self.model_idx_map_short['y'][self.base_model_type][self.base_model_id], self.model_idx_map_short['uv'][self.base_model_type][self.base_model_id]
        else:
            return self.model_idx_map_long['y'][self.base_model_id], self.model_idx_map_long['uv'][self.base_model_id]  
    
    @property
    def map_idx_long(self) -> int:
        return self.model_idx_map_long['y'][self.base_model_id], self.model_idx_map_long['uv'][self.base_model_id]  
    
    
    def _params_loaded(self) -> None:
        self.model_state_per_tile = list()
        self.model_state = None
        
        
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

        def decode_tile(tile_idx, tile):
            use_YUV = [0]*3
            model_selection = [0]*3
            model_selection_Y = model_selection_UV = 0


            for i in range(3):
                use_YUV[i] = int(ec.decode([1], 1, name=f'icci_use[{i}]'))
            if any(use_YUV):
                self.use_short_list = int(ec.decode([1], 1, name='icci_use_shortList'))

                map_y, map_uv = self.map_idx
                
                if use_YUV[0]:                    
                    model_selection_Y = int(ec.decode([1], self.models_count, name=f'icci_model_signalled_idx[{tile_idx}][Y]'))
                    model_selection[0] = map_y[model_selection_Y] + 1

                if any(use_YUV[1:]):
                    model_selection_UV = int(ec.decode([1], self.models_count, name=f'icci_model_signalled_idx[{tile_idx}][UV]'))
                    if use_YUV[1]:
                        model_selection[1] = map_uv[model_selection_UV] + 1
                    if use_YUV[2]:
                        model_selection[2] = map_uv[model_selection_UV] + 1
                    
            logger.debug(f'PPS (dec): tile {tile_idx, tile} | '
                         f'use_YUV: {use_YUV} | '
                         f'use_short_list: {self.use_short_list} | '
                         f'model_selection_Y: {model_selection_Y} | '
                         f'model_selection_UV: {model_selection_UV} | '
                         f'model_IDs: {model_selection}')

            return model_selection
        
        
        if tile_manager.is_enabled():
            for tile_idx, tile in enumerate(tile_manager.image_tiles):
                    self.model_state_per_tile.append(decode_tile(tile_idx,tile))
        else:
            self.model_state_per_tile.append(decode_tile(0,0))

    def encode_header(self, ec: HeaderCoder):
        """
        Encodes model selection into the bitstream using the Entropy
        Coder Engine.

        Args:
            ec: Entropy Coder Engine
        """

        tile_manager = self.__dict__['tile_manager']
        logger = self.logger

        def encode_tile(tile_idx, tile):
            model_selection = self.model_state_per_tile[tile_idx]
            map_y, map_uv = self.map_idx

            use_YUV = list(map(lambda x: int(bool(x)), model_selection))
            use_short_list = self.use_short_list
            model_selection_Y = None
            model_selection_UV = None
            
            for i in range(3):
                ec.encode(use_YUV[i], 1, name=f'icci_use[{i}]')
            
            if any(use_YUV):
                ec.encode(use_short_list, 1, name='icci_use_shortList')
                if use_YUV[0]:
                    model_idx = model_selection[0] - 1
                    model_selection_Y = map_y.index(model_idx)
                    ec.encode(model_selection_Y,
                            self.models_count,
                            name=f'icci_model_signalled_idx[{tile_idx}][Y]')
                
                if any(use_YUV[1:]):
                    model_idx = (model_selection[1] or model_selection[2]) - 1
                    model_selection_UV = map_uv.index(model_idx)

                    ec.encode(model_selection_UV,
                            self.models_count,
                            name=f'icci_model_signalled_idx[{tile_idx}][UV]')
                    
                         
            logger.debug(f'PPS (enc): tile {tile_idx, tile} | '
                         f'use_YUV: {use_YUV} | '
                         f'use_short_list: {use_short_list} | '
                         f'model_selection_Y: {model_selection_Y} | '
                         f'model_selection_UV: {model_selection_UV} | '
                         f'model_IDs: {model_selection}')
            

        if tile_manager.is_enabled():
            for tile_idx, tile in enumerate(tile_manager.image_tiles):
                encode_tile(tile_idx, tile)    
        else:
            encode_tile(0, 0)   