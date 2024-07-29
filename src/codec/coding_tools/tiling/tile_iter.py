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

import torch

##
from src.codec.common import Image, tiling


class TileIterObject:
    def __init__(self, tile_manager, input_data: torch.Tensor, output_data,
                 input_tiles: tiling.TileGrid, output_tiles: tiling.TileGrid, exclude_boundaries,
                 idx_x, idx_y):
        self.tile_manager = tile_manager
        self.input_data = input_data
        self.output_data = output_data
        self.exclude_boundaries = exclude_boundaries
        self.ad = tiling.assign_data2image if isinstance(output_data,
                                                         Image) else tiling.assign_data
        self.idx = input_tiles.columns * idx_y + idx_x
        self._input_tile = input_tiles.tiles[idx_y][idx_x]
        self._output_tile = output_tiles.tiles[idx_y][idx_x]

    @property
    def input_tile(self) -> tiling.Area:
        return self._input_tile

    @property
    def output_tile(self) -> tiling.Area:
        return self._output_tile

    def input_shape(self) -> torch.Size:
        return torch.Size([self.input_tile.size.height, self.input_tile.size.width])

    def output_shape(self) -> torch.Size:
        return torch.Size([self.output_tile.size.height, self.output_tile.size.width])

    def get_data(self) -> torch.Tensor:
        return tiling.get_data(self.input_data, self.input_tile)

    def get_idx(self) -> int:
        return self.idx

    def assign_data(self, data: torch.Tensor) -> None:
        if self.exclude_boundaries:
            # assign tile, but leave out features at boundaries by not using
            # part of overlapped regions
            assigned_tile, assigned_tile_rel_to_overlap = self.tile_manager.get_core_of_overlapping_image_tile(
                self.output_tile)
            assigned_tile_data = tiling.get_data(data, assigned_tile_rel_to_overlap)
        else:
            assigned_tile_data = data
            assigned_tile = self.output_tile

        self.ad(self.output_data, assigned_tile, assigned_tile_data)
