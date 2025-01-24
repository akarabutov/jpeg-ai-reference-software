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
from typing import Tuple

import torch

from src.codec.common.tiling import Area, TileGrid, Position
from src.codec.entropy_coding import HeaderCoder

from ..interfaces import CoderEngine
##
from .params import TilingParams
from .tile_iter import TileIterObject
from copy import deepcopy

class TileManager(CoderEngine):
    """Class for creating TileGrid, that is used in various places of the project
    (enc/dec, icci, RDLR).
    """
    def __init__(self, alignment_size: int, latent_downscale_factor_y: int, **kwards):
        super(TileManager, self).__init__(has_enabled_flag=True, **kwards)
        self.__params_tiling = TilingParams(**kwards)

        self.image_shape: torch.Size
        self.latent_shape: torch.Size
        self.alignment_size: int = alignment_size
        self.latent_downscale_factor_y: int = latent_downscale_factor_y
        self.latent_downscale_factor_psi: int = self.latent_downscale_factor_y * 2
        self.latent_downscale_factor_z: int = self.latent_downscale_factor_y * 4
        self.tile_height: int
        self.tile_width: int
        self.tile_size: int = 0
        # self.overlap: int
        self.minimum_tile_size: int  # for adjusting tiles at boundary
        self.image_tiles: TileGrid
        self.image_tiles_entropy_coding: TileGrid
        self.latent_tiles: TileGrid
        self.latent_tiles_entropy_coding: TileGrid
        self.numSamplesPerTile: int
        self.numSamplesTileOverlap: int

    def calc_latent_shape(self, image_shape: torch.Size, chs: int) -> torch.Size:
        """Calculate size of latent space from size of signal space and alignment size.

        Args:
            image_shape (torch.Size): size of image in signal space
            chs (int): number of channels generated size should have.

        Returns:
            torch.Size: Size of latent space.
        """
        latent_height = math.ceil(image_shape[-2] / self.alignment_size)
        latent_width = math.ceil(image_shape[-1] / self.alignment_size)
        return (image_shape[0], chs, latent_height, latent_width)

    # only used by rdlr, do not care for now
    # def set_alignment_size(self, value: int) -> None:
    #     """store the alignment size.

    #     Args:
    #         value (int): alignment size that will be stored.
    #     """
    #     self.alignment_size = value
    #     self.latent_downscale_factor_y = value

    def encode_header(self, ec: HeaderCoder) -> None:
        """Function for encoding header fields for tiling. I.e. tile size and overlap.

        Args:
            ec (HeaderCoder): entropy coding module that is used for encoding the data.
        """
        assert self.tile_size & 0xF == 0
        assert self.numSamplesTileOverlap & 0xF == 0
        ec.encode(self.tile_size // 16, bits_count=8, name='synthesis_tile_size')
        ec.encode(self.numSamplesTileOverlap // 16, bits_count=5, name='synthesis_tile_overlap')

    def decode_header(self, ec: HeaderCoder) -> None:
        """Function for decoding header fields for tiling. I.e. tile size and overlap.

        Args:
            ec (HeaderCoder): entropy coding module that is used for decoding the data.
        """

        self.tile_size = int(
            ec.decode([1], bits_count=8, name='synthesis_tile_size')) * 16
        self.numSamplesTileOverlap = int(ec.decode([1], bits_count=5, name='synthesis_tile_overlap'))*16

    def setup_tiles_enc(self,
                        image_shape: torch.Size,
                        latent_shape: torch.Size,
                        latent_shape_psi: torch.Size,
                        latent_shape_z: torch.Size,
                        minimum_tile_size: int = None):
        """Function for initializing the TileGrid used for the image at the encoder side.
        Tiles are stored in the class
        variables `image_tiles` and `latent_tiles`. This function determines from image size
        whether tiling is required and en/disabled is accordingly. It also selects tile size.
        Tile size is constrained by configuration (to meet memory requirements) as well as
        image height/width and alignment size.
        If tiling is not enabled the generated TileGrids have only one tile the same size as the
        image.

        Args:
            image_shape (torch.Size): Size of the image in signal space to generate tiles for.
            latent_shape (torch.Size): Size in latent space domain to generate tiles for y. If it is None
            no `latent_tiles` are initialized.
            latent_shape_z (torch.Size): Size in latent space domain to generate tiles for z. If it is None
            no `latent_tiles` are initialized.
            minimum_tile_size (int, optional): If given, tiles need to have at least this
            width/height. Only used at bottom/right image boundary, where tile size can differ
            from regular size. If a tile here is smaller than `minimum_tile_size` it is
            enlarged by shrinking its neighboring tile. Defaults to None.
        """

        img_height, img_width = image_shape[2:]
        self.set_enable(self.numSamplesPerTile != -1
                        and self.numSamplesPerTile < img_height * img_width)
        if self.is_enabled():
            self.tile_size = math.floor(math.sqrt(
                self.numSamplesPerTile))  # less samples than numSamplesPerTile
            self.tile_size = math.ceil(self.tile_size /
                                self.alignment_size) * self.alignment_size  # multiple of alignment_size
            assert self.numSamplesTileOverlap % self.alignment_size == 0, f'Overlap for tiles should be multiple of alignment_size ({self.alignment_size})'

        self.setup_tiles_dec(image_shape, latent_shape, latent_shape_psi, latent_shape_z, minimum_tile_size)

    def setup_tiles_dec(self,
                        image_shape: torch.Size,
                        latent_shape: torch.Size,
                        latent_shape_psi: torch.Size,
                        latent_shape_z: torch.Size,
                        minimum_tile_size: int = None):
        """Function for initializing the TileGrid used for the image at the decoder side.
        Tiles are stored in the class
        variables `image_tiles` and `latent_tiles`. If tiling is not enabled the generated
        TileGrids have only one tile the same size as the image.

        Args:
            image_shape (torch.Size): Size of the image in signal space to generate tiles for.
            latent_shape (torch.Size): Size in latent space domain to generate tiles for y. If it is None
            no `latent_tiles` are initialized.
            latent_shape_z (torch.Size): Size in latent space domain to generate tiles for z. If it is None
            no `latent_tiles` are initialized.
            minimum_tile_size (int, optional): If given, tiles need to have at least this
            width/height. Only used at bottom/right image boundary, where tile size can differ
            from regular size. If a tile here is smaller than `minimum_tile_size` it is
            enlarged by shrinking its neighboring tile. Defaults to None.
        """
        self.image_shape = image_shape
        self.latent_shape = latent_shape
        self.latent_shape_psi = latent_shape_psi
        self.latent_shape_z = latent_shape_z
        self.minimum_tile_size = minimum_tile_size
        img_height, img_width = image_shape[2:]

        if self.is_enabled():
            self.tile_size_y = self.tile_size // self.latent_downscale_factor_y
            self.tile_size_psi = self.tile_size // self.latent_downscale_factor_psi
            self.tile_size_z = self.tile_size // self.latent_downscale_factor_z

        if self.is_enabled():
            self.tile_height = min(self.tile_size, img_height)
            self.tile_width = min(self.tile_size, img_width)
        else:
            self.tile_height = img_height
            self.tile_width = img_width


        if not self.is_enabled():
            # no tiles: i.e. treat whole image as a single tile
            _, _, height, width = image_shape
            self.image_tiles = TileGrid(1, 1, [[Area.from_x_y_width_height(0, 0, width, height)]])
            self.image_tiles_entropy_coding = TileGrid(1, 1, [[Area.from_x_y_width_height(0, 0, width, height)]])
        else:
            if not hasattr(self, 'region_residual_in_its_own_substream_flag'):
                self.region_residual_in_its_own_substream_flag = False
            self.image_tiles = self._init_image_tiles_with_overlap( 
            ) if (self.numSamplesTileOverlap and not self.region_residual_in_its_own_substream_flag) else self._init_image_tiles()
            if self.minimum_tile_size is not None:
                self.minimum_tile_size = minimum_tile_size
                self._adjust_boundary_tiles()
            self.image_tiles_entropy_coding = self._init_image_tiles_with_overlap( )

        if self.latent_shape is not None:
            self.latent_tiles = self.init_latent_tiles(self.image_tiles, self.latent_shape, self.latent_downscale_factor_y)
            self.latent_tiles_entropy_coding = self.init_latent_tiles(self.image_tiles_entropy_coding, self.latent_shape, self.latent_downscale_factor_y)
            self.latent_tiles_psi = self.init_latent_tiles(self.image_tiles, self.latent_shape_psi, self.latent_downscale_factor_psi)
            self.latent_tiles_z = self.init_latent_tiles(self.image_tiles, self.latent_shape_z, self.latent_downscale_factor_z)


    def setup_tiles_dec_from_region_tile_manager(self, region_tile_manager,
                        image_shape: torch.Size,
                        latent_shape: torch.Size,
                        latent_shape_psi: torch.Size,
                        latent_shape_z: torch.Size,
                        minimum_tile_size: int = None,
                        alighment_size: int=16):
        """Function for initializing the TileGrid used for the image at the decoder side.
        Tiles are stored in the class
        variables `image_tiles` and `latent_tiles`. If tiling is not enabled the generated
        TileGrids have only one tile the same size as the image.

        Args:
            image_shape (torch.Size): Size of the image in signal space to generate tiles for.
            latent_shape (torch.Size): Size in latent space domain to generate tiles for y. If it is None
            no `latent_tiles` are initialized.
            latent_shape_z (torch.Size): Size in latent space domain to generate tiles for z. If it is None
            no `latent_tiles` are initialized.
            minimum_tile_size (int, optional): If given, tiles need to have at least this
            width/height. Only used at bottom/right image boundary, where tile size can differ
            from regular size. If a tile here is smaller than `minimum_tile_size` it is
            enlarged by shrinking its neighboring tile. Defaults to None.
        """
        assert alighment_size in [8, 16]
        self.image_shape = image_shape
        self.latent_shape = latent_shape
        assert latent_shape == region_tile_manager.latent_shape
        self.latent_shape_psi = latent_shape_psi
        self.latent_shape_z = latent_shape_z
        self.minimum_tile_size = minimum_tile_size
        img_height, img_width = image_shape[2:]

        self.enabled = True
        self.image_tiles = deepcopy(region_tile_manager.img_tiles)
        self.image_tiles_entropy_coding = deepcopy(region_tile_manager.img_tiles_withoutExtend)

        is_chroma = 1 if alighment_size == 8 else 0
        if is_chroma:
            scaled_tiles = [[ im_tile.upscale(2) for im_tile in tiles_row] for tiles_row in self.image_tiles.tiles]
            self.image_tiles = TileGrid( self.image_tiles.rows, self.image_tiles.columns, scaled_tiles)
            scaled_tiles = [[ im_tile.upscale(2) for im_tile in tiles_row] for tiles_row in self.image_tiles_entropy_coding.tiles]
            self.image_tiles_entropy_coding = TileGrid( self.image_tiles_entropy_coding.rows, self.image_tiles_entropy_coding.columns, scaled_tiles)

        self.latent_tiles =  deepcopy(region_tile_manager.latent_tiles_withExtend)
        self.latent_tiles_entropy_coding =  deepcopy(region_tile_manager.latent_tiles)
        self.latent_tiles_psi =  deepcopy(region_tile_manager.psi_tiles_withExtend)
        self.latent_tiles_z =  deepcopy(region_tile_manager.z_tiles)

        #     self.tile_size_y = self.tile_size // self.latent_downscale_factor_y
        #     self.tile_size_psi = self.tile_size // self.latent_downscale_factor_psi
        #     self.tile_size_z = self.tile_size // self.latent_downscale_factor_z

        # if self.is_enabled():
        #     self.tile_height = min(self.tile_size, img_height)
        #     self.tile_width = min(self.tile_size, img_width)
        # else:
        #     self.tile_height = img_height
        #     self.tile_width = img_width



                # self.psi_tiles_withExtend,
                # self.psi_tiles,
                # self.latent_tiles_withExtend,
                # self.latent_tiles,
                # self.z_tiles,
                # self.z_tiles_withoutExtend,
                # self.img_tiles,
                # self.img_tiles_withoutExtend,
                # self.flatten_list):


        # if self.latent_shape is not None:
        #     self.latent_tiles = self.init_latent_tiles(self.image_tiles, self.latent_shape, self.latent_downscale_factor_y, 0)
        #     self.latent_tiles_entropy_coding = self.init_latent_tiles(self.image_tiles_entropy_coding, self.latent_shape, self.latent_downscale_factor_y, 0)
        #     self.latent_tiles_psi = self.init_latent_tiles(self.image_tiles, self.latent_shape_psi, self.latent_downscale_factor_psi, 0)
        #     self.latent_tiles_z = self.init_latent_tiles(self.image_tiles, self.latent_shape_z, self.latent_downscale_factor_z, 0)

    def _add_overlap(self, tile: Area, latent_tile_grid_in_image_domain: TileGrid, overlap_amount: int) -> Area:
        """
        When multiple tiles inside an independent region, add overlapping to the boundaries.
        Args:
            tile:
            latent_tiles_in_image_domain:

        Returns:

        """
        _, _, image_hh, image_ww = self.image_shape
        _, _, latent_hh, latent_ww = self.latent_shape

        if int(math.ceil(image_hh / 8)) == latent_hh:  # encoder, chroma
            scaled_tiles = [[tile.downscale(2) for tile in row] for row in latent_tile_grid_in_image_domain.tiles]
            latent_tile_grid_in_image_domain = TileGrid(latent_tile_grid_in_image_domain.rows,
                                                        latent_tile_grid_in_image_domain.columns,
                                                        scaled_tiles)
        y_start, x_start = tile.position.y, tile.position.x
        y_end, x_end = tile.bottom_right().y, tile.bottom_right().x
        tile_height, tile_width = tile.size.height, tile.size.width
        rows, cols = latent_tile_grid_in_image_domain.rows, latent_tile_grid_in_image_domain.columns
        for i in range(rows):
            for j in range(cols):
                region = latent_tile_grid_in_image_domain.get_tile(i, j)
                if region.contains(Position(x_start, y_start)) and region.contains(Position(x_end, y_end)):
                    if y_start > region.position.y:
                        y_start -= overlap_amount
                        tile_height += overlap_amount
                    if x_start > region.position.x:
                        x_start -= overlap_amount
                        tile_width += overlap_amount
                    if y_end < region.bottom_right().y:
                        tile_height += min(overlap_amount, image_hh - (y_start + tile_height))
                    if x_end < region.bottom_right().x:
                        tile_width += min(overlap_amount, image_ww - (x_start + tile_width))
        new_tile = Area.from_x_y_width_height(x_start, y_start, tile_width, tile_height)
        return new_tile

    def _init_image_tiles(self) -> TileGrid:
        """Split image into tiles of given width and height. All tiles except last row and last
        column need to have width and height, which are a multiple of `alignment_size`.
        Tiles are returned via a instance of `TileGrid`.

        Returns:
            TileGrid: The image partitioned into tiles. No actual data, just locations are stored
            in this.
        """
        _, _, image_hh, image_ww = self.image_shape
        if self.tile_height == self.tile_size:  # exception for single row/column of tiles
            assert self.tile_height % self.alignment_size == 0, 'Tile alignment error. Requested tile size not multiple of alignment size!'
        if self.tile_width == self.tile_size: # exception for single row/column of tiles
            assert self.tile_width % self.alignment_size == 0, 'Tile alignment error. Requested tile size not multiple of alignment size!'
        assert self.tile_height <= image_hh, 'Tile can not be larger than image!'
        assert self.tile_width <= image_ww, 'Tile can not be larger than image!'

        image_tiles = []
        for tile_start_y in range(0, image_hh, self.tile_height):
            image_tile_row = []
            for tile_start_x in range(0, image_ww, self.tile_width):
                height = min(self.tile_height, image_hh - tile_start_y)
                width = min(self.tile_width, image_ww - tile_start_x)
                tile_no_overlap = Area.from_x_y_width_height(tile_start_x, tile_start_y, width, height)
                tile_with_overlap = self._add_overlap(tile_no_overlap, self.owner.common_modules.tile_manager_hd.image_tiles, self.numSamplesTileOverlap // 2)
                image_tile_row.append(tile_with_overlap)
            image_tiles.append(image_tile_row)
        rows = len(image_tiles)
        columns = len(image_tiles[0])

        return TileGrid(rows, columns, image_tiles)

    def _init_image_tiles_with_overlap(self) -> TileGrid:
        """Split image into tiles of given width and height. All tiles except last row and last
        column need to have width and height, which are a multiple of `alignment_size`. Tiles will
        overlap by the given `overlap`, which also need to be a multiple of alignment size or zero.
        Tiles are returned via a instance of `TileGrid`.

        Returns:
            TileGrid: The image partitioned into tiles. No actual data, just locations are stored
            in this.
        """
        _, _, image_hh, image_ww = self.image_shape
        if self.tile_height == self.tile_size:  # exception for single row/column of tiles
            assert self.tile_height % self.alignment_size == 0, 'Tile alignment error. Requested tile size not multiple of alignment size!'
        if self.tile_width == self.tile_size: # exception for single row/column of tiles
            assert self.tile_width % self.alignment_size == 0, 'Tile alignment error. Requested tile size not multiple of alignment size!'
        assert self.numSamplesTileOverlap % self.alignment_size == 0, 'Tile overlap error. Requested overlap size not multiple of alignment size!'
        assert self.tile_height <= image_hh, 'Tile can not be larger than image!'
        assert self.tile_width <= image_ww, 'Tile can not be larger than image!'

        image_tiles = []
        for tile_start_y in range(0, image_hh - self.numSamplesTileOverlap,
                                  self.tile_height - self.numSamplesTileOverlap):
            image_tile_row = []
            for tile_start_x in range(0, image_ww - self.numSamplesTileOverlap,
                                      self.tile_width - self.numSamplesTileOverlap):
                height = min(self.tile_height, image_hh - tile_start_y)
                width = min(self.tile_width, image_ww - tile_start_x)
                tile = Area.from_x_y_width_height(tile_start_x, tile_start_y, width, height)
                image_tile_row.append(tile)
            image_tiles.append(image_tile_row)
        rows = len(image_tiles)
        columns = len(image_tiles[0])

        return TileGrid(rows, columns, image_tiles)

    def _get_latent_tile_from_image_tile(self, image_tile: Area, latent_shape: torch.Size, latent_downscale_factor: int) -> Area:
        """From an image tile obtain the matching feature tile. Images tile except if in last
        row and last column needs to have
        width and height, which are a multiple of `alignment_size`.

        Args:
            image_tile (Area): tile of an image.
            latent_shape (torch.Size): shape of the latent tensor 
            latent_downscale_factor (int): Factor by which latent tile is smaller than given image tile

        Returns:
            Area: tile of features.
        """
        if image_tile is None:
            return None

        # tile_size_latent = self.tile_size // latent_downscale_factor

        _, _, latent_hh, latent_ww = latent_shape
        lat_tile_start_y = image_tile.position.y // latent_downscale_factor
        lat_tile_start_x = image_tile.position.x // latent_downscale_factor
        if image_tile.size.height % latent_downscale_factor:
            assert math.ceil(
                (image_tile.position.y + image_tile.size.height) / latent_downscale_factor
            ) >= latent_hh, 'Tile of not aligned size which is not at image border!'
            height = math.ceil(image_tile.size.height / latent_downscale_factor)
        else:
            height = image_tile.size.height // latent_downscale_factor
        if image_tile.size.width % latent_downscale_factor:
            assert math.ceil(
                (image_tile.position.x + image_tile.size.width) / latent_downscale_factor
            ) >= latent_ww, 'Tile of not aligned size which is not at image border!'
            width = math.ceil(image_tile.size.width / latent_downscale_factor)
        else:
            width = image_tile.size.width // latent_downscale_factor

        return Area.from_x_y_width_height(lat_tile_start_x, lat_tile_start_y, width, height)

    def init_latent_tiles(self, image_tiles: TileGrid, latent_shape: torch.Size, latent_downscale_factor: int) -> TileGrid:
        """From a `TileGrid` of image tiles obtain the matching `TileGrid` of feature tiles.
        All images tiles except last row and last column need to have
        width and height, which are a multiple of `alignment_size`.
        Tiles are returned via a instance of `TileGrid`.

        Args:
            image_tiles (TileGrid): tile grid for which to generate the corresponding latent
            tile grid.
            latent_shape (torch.Size): shape of the latent tensor 
            latent_downscale_factor (int): Factor by which latent tile is smaller than given image tile

        Returns:
            TileGrid: The features partitioned into tiles. No actual data,
            just locations are stored in this.
        """

        latent_tiles = [[self._get_latent_tile_from_image_tile(im_tile, latent_shape, latent_downscale_factor) for im_tile in tiles_row]
                        for tiles_row in image_tiles.tiles]
        return TileGrid(image_tiles.rows, image_tiles.columns, latent_tiles)

    def get_core_of_overlapping_image_tile(self, image_tile: Area) -> Tuple[Area, Area]:
        """For tiles with overlap regins, find the inner region without overlap.
        Assign tile, but leave out features/image at boundaries by not using
        part of overlapped regions. The used overlap differs depending on whether
        a tile is at an image boundary.

        Args:
            image_tile (Area): tile for which to retrieve tile without overlap.
            can be image or latent tile

        Returns:
            Tuple[Area, Area]: First: inner part of overlapping tile,
            i.e. wihtout the overlap. Second: inner tile relative to the overlap.
        """

        return self._get_core_of_overlapping_tile(image_tile, image_tile, 1, None)

    def get_core_of_overlapping_latent_tile(self, latent_tile: Area,
                                            corresponding_image_tile: Area,
                                            tile_manager_hd: "TileManager") -> Tuple[Area, Area]:
        """For tiles with overlap regions, find the inner region without overlap.
        Assign tile, but leave out features/image at boundaries by not using
        part of overlapped regions. The used overlap differs depending on
        whether a tile is at an image boundary.

        Args:
            latent_tile (Area): tile for which to retrieve tile without overlap.
            can be image or latent tile
            corresponding_image_tile (Area): the corresponding image tile.
            If `latent_tile` is image tile, the same as this tile.

        Returns:
            Tuple[Area, Area]: First: inner part of overlapping tile,
            i.e. wihtout the overlap. Second: inner tile relative to the overlap.
        """

        return self._get_core_of_overlapping_tile(latent_tile, corresponding_image_tile,
                                                  self.latent_downscale_factor_y, tile_manager_hd)

    def get_core_of_overlapping_latent_tile_psi(self, latent_tile: Area,
                                            corresponding_image_tile: Area,
                                            tile_manager_hd: "TileManager") -> Tuple[Area, Area]:
        """For tiles with overlap regions, find the inner region without overlap.
        Assign tile, but leave out features/image at boundaries by not using
        part of overlapped regions. The used overlap differs depending on
        whether a tile is at an image boundary.

        Args:
            latent_tile (Area): tile for which to retrieve tile without overlap.
            can be image or latent tile
            corresponding_image_tile (Area): the corresponding image tile.
            If `latent_tile` is image tile, the same as this tile.

        Returns:
            Tuple[Area, Area]: First: inner part of overlapping tile,
            i.e. wihtout the overlap. Second: inner tile relative to the overlap.
        """

        return self._get_core_of_overlapping_tile(latent_tile, corresponding_image_tile,
                                                  self.latent_downscale_factor_psi, tile_manager_hd)

    def get_core_of_overlapping_latent_tile_z(self, latent_tile: Area,
                                            corresponding_image_tile: Area,
                                            tile_manager_hd: "TileManager") -> Tuple[Area, Area]:
        """For tiles with overlap regions, find the inner region without overlap.
        Assign tile, but leave out features/image at boundaries by not using
        part of overlapped regions. The used overlap differs depending on
        whether a tile is at an image boundary.

        Args:
            latent_tile (Area): tile for which to retrieve tile without overlap.
            can be image or latent tile
            corresponding_image_tile (Area): the corresponding image tile.
            If `latent_tile` is image tile, the same as this tile.

        Returns:
            Tuple[Area, Area]: First: inner part of overlapping tile,
            i.e. wihtout the overlap. Second: inner tile relative to the overlap.
        """

        return self._get_core_of_overlapping_tile(latent_tile, corresponding_image_tile,
                                                  self.latent_downscale_factor_z, tile_manager_hd)

    def _get_core_of_overlapping_tile(self, image_or_latent_tile: Area,
                                      corresponding_image_tile: Area,
                                      alignment_size: int,
                                      tile_manager_hd: "TileManager") -> Tuple[Area, Area]:
        """For tiles with overlap regins, find the inner region without overlap.
        Assign tile, but leave out features/image at boundaries by not using
        part of overlapped regions. The used overlap differs depending on
        whether a tile is at an image boundary.

        Args:
            image_or_latent_tile (Area): tile for which to retrieve tile without overlap.
            can be image or latent tile
            corresponding_image_tile (Area): the corresponding image tile.
            If `image_or_latent_tile` is image tile, the same as this tile.
            alignment_size (int): Size to which tile boundaries are aligned.
        
        Returns:
            Tuple[Area, Area]: First: inner part of overlapping tile, i.e. wihtout the overlap.
            Second: inner tile relative to the overlap.
        """
        if hasattr(self, "region_partitioning_flag") and self.get_owner_param('region_partitioning_flag') and self.get_owner_param('region_residual_in_its_own_substream_flag'):
            if tile_manager_hd is None:
                tile_manager_hd = self.get_owner_param("common_modules").tile_manager_hd
            region_tiles = tile_manager_hd.image_tiles
            if alignment_size == self.latent_downscale_factor_z:
                region_tiles = tile_manager_hd.latent_tiles_z
            if alignment_size == self.latent_downscale_factor_y:
                region_tiles = tile_manager_hd.latent_tiles
            overlap_amount = self.numSamplesTileOverlap // 2 // alignment_size
            y_start, x_start = image_or_latent_tile.position.y, image_or_latent_tile.position.x
            y_end, x_end = image_or_latent_tile.bottom_right().y, image_or_latent_tile.bottom_right().x
            core_height, core_width = image_or_latent_tile.size.height, image_or_latent_tile.size.width
            overlap_top, overlap_bottom = 0, 0
            overlap_left, overlap_right = 0, 0
            for i in range(region_tiles.rows):
                for j in range(region_tiles.columns):
                    region = region_tiles.get_tile(i, j)
                    if region.contains(Position(x_start, y_start)) and region.contains(Position(x_end, y_end)):
                        if y_start > region.position.y:
                            overlap_top += overlap_amount
                            y_start += overlap_amount
                            core_height -= overlap_amount
                        if x_start > region.position.x:
                            overlap_left += overlap_amount
                            x_start += overlap_amount
                            core_width -= overlap_amount
                        core_height = min(core_height, int(math.ceil(self.tile_height / alignment_size)))
                        core_width = min(core_width, int(math.ceil(self.tile_width / alignment_size)))

            core_tile = Area.from_x_y_width_height(x_start, y_start, core_width, core_height)
            core_tile_relative_to_overlap = Area.from_x_y_width_height(
                overlap_left,
                overlap_top,
                core_width,
                core_height)
            return core_tile, core_tile_relative_to_overlap

        _, _, img_height, img_width = self.image_shape
        overlap_left = 0 if corresponding_image_tile.top_left(
        ).x == 0 else self.numSamplesTileOverlap // 2 // alignment_size
        overlap_top = 0 if corresponding_image_tile.top_left(
        ).y == 0 else self.numSamplesTileOverlap // 2 // alignment_size
        overlap_right = 0 if corresponding_image_tile.bottom_right(
        ).x + 1 >= img_width else self.numSamplesTileOverlap // 2 // alignment_size
        overlap_bottom = 0 if corresponding_image_tile.bottom_right(
        ).y + 1 >= img_height else self.numSamplesTileOverlap // 2 // alignment_size
        core_tile = Area.from_x_y_width_height(
            image_or_latent_tile.position.x + overlap_left,
            image_or_latent_tile.position.y + overlap_top,
            image_or_latent_tile.size.width - overlap_right - overlap_left,
            image_or_latent_tile.size.height - overlap_bottom - overlap_top,
        )
        core_tile_relative_to_overlap = Area.from_x_y_width_height(
            overlap_left,
            overlap_top,
            core_tile.size.width,
            core_tile.size.height,
        )

        return core_tile, core_tile_relative_to_overlap

    def _adjust_boundary_tiles(self) -> TileGrid:
        """Processes last two rows and columns of a `TileGrid`. If last row/colum is smaller in
        height/width than `minimum_tile_size` it is increased and the increase is taken from
        the second-to-last row/column. Alignment `alignment_size` will be preserved.
        Only application thus far is for MS-SSIM, which requires size of 32,
        but alignment size required by NN is lower/16.

        Returns:
            TileGrid: The modified tile grid, where last row/column may have been enlarged.
        """
        assert self.minimum_tile_size % self.alignment_size == 0, f'Minimum tile size ({self.minimum_tile_size}) needs to be multiple of alignment size ({self.alignment_size})!'

        bottom_right_tile = self.image_tiles.get_tile(-1, -1)

        # check width of last tile column
        if bottom_right_tile.size.width < self.minimum_tile_size:
            # adjust. simply take last two tile columns. make last one larger by reducing second last one a bit

            for row_idx in range(self.image_tiles.rows):
                second_to_last_tile = self.image_tiles.tiles[row_idx][-2]
                last_tile = self.image_tiles.tiles[row_idx][-1]
                self.image_tiles.tiles[row_idx][-2] = Area.from_position_width_height(
                    second_to_last_tile.position,
                    second_to_last_tile.size.width - self.minimum_tile_size,
                    second_to_last_tile.size.height
                )
                self.image_tiles.tiles[row_idx][-1] = Area.from_x_y_width_height(
                    last_tile.position.x - self.minimum_tile_size,
                    last_tile.position.y,
                    last_tile.size.width + self.minimum_tile_size,
                    last_tile.size.height
                )

        # check height of last tile row
        if bottom_right_tile.size.height < self.minimum_tile_size:
            # adjust. simply take last two tile rows. make last one larger by reducing second last one a bit
            for col_idx in range(self.image_tiles.columns):
                second_to_last_tile = self.image_tiles.tiles[-2][col_idx]
                last_tile = self.image_tiles.tiles[-1][col_idx]
                self.image_tiles.tiles[-2][col_idx] = Area.from_position_width_height(
                    second_to_last_tile.position,
                    second_to_last_tile.size.width,
                    second_to_last_tile.size.height - self.minimum_tile_size
                )
                self.image_tiles.tiles[-1][col_idx] = Area.from_x_y_width_height(
                    last_tile.position.x,
                    last_tile.position.y - self.minimum_tile_size,
                    last_tile.size.width,
                    last_tile.size.height + self.minimum_tile_size
                )

    def get_iter_over_tiles(self, input_data: torch.Tensor, output_data: torch.Tensor):
        for idx_y in range(self.latent_tiles.rows):
            for idx_x in range(self.latent_tiles.columns):
                yield TileIterObject(self, input_data, output_data, self.latent_tiles,
                                     self.image_tiles, self.is_enabled(), idx_x, idx_y)

    def __len__(self):
        return self.latent_tiles.rows * self.latent_tiles.columns



class TileManagerHyper(CoderEngine):
    """Class for creating TileGrid, that is used in various places of the project
    (enc/dec, icci, RDLR).
    """

    def __init__(self, **kwards):
        super(TileManagerHyper, self).__init__(has_enabled_flag=True, **kwards)

        self.latent_shape: torch.Size
        self.latent_downscale_factor: int
        self.z_downscale_factor: int

        self.tile_height: int = 0
        self.tile_width: int = 0
        # self.overlap: int

        self.z_tiles: TileGrid
        self.z_tiles_withoutExtend: TileGrid
        self.latent_tiles: TileGrid
        self.latent_tiles_withExtend: TileGrid
        self.img_tiles_withoutExtend: TileGrid

    def calculate_region_coordinates(self, height, width, depth):
        if self.region_partitioning_flag == 0:
            self.horizontal_reg_coords_start = [0]
            self.horizontal_reg_coords_end = [math.ceil(width / (2 ** depth))]
            self.vertical_reg_coords_start = [0]
            self.vertical_reg_coords_end = [math.ceil(height / (2 ** depth))]
        else:
            verRegionSize = math.floor(math.floor((height + 127) / 128) / self.numVerRegions) * 128
            horRegionSize = math.floor(math.floor((width + 127) / 128) / self.numHorRegions) * 128
            width_feature = math.ceil(width / (2 ** depth))
            height_feature = math.ceil(height / (2 ** depth))
            #width_step = math.ceil(horRegionSize / (2 ** depth))
            #height_step = math.ceil(verRegionSize / (2 ** depth))
            self.horizontal_reg_coords_start = []
            self.horizontal_reg_coords_end = []
            self.vertical_reg_coords_start = []
            self.vertical_reg_coords_end = []
            for x in range(0, self.numHorRegions):
                self.horizontal_reg_coords_start.append(math.floor(x * horRegionSize / (2 ** depth)))
                #(x * width_step)
                if x < self.numHorRegions - 1:
                    self.horizontal_reg_coords_end.append(math.floor((x+1) * horRegionSize / (2 ** depth)))
                    #((x + 1) * width_step)
                else:
                    self.horizontal_reg_coords_end.append(width_feature)
            for y in range(0, self.numVerRegions):
                self.vertical_reg_coords_start.append(math.floor(y * verRegionSize / (2 ** depth)))
                #(y * height_step)
                if y < self.numVerRegions - 1:
                    self.vertical_reg_coords_end.append(math.floor((y+1) * verRegionSize / (2 ** depth)))
                    #((y + 1) * height_step)
                else:
                    self.vertical_reg_coords_end.append(height_feature)
            assert (self.numHorRegions == len(self.horizontal_reg_coords_start))
            assert (self.numVerRegions == len(self.vertical_reg_coords_start))

    def _init(self, y_shape, alignment_size=16) -> None:
        """store the alignment size.

        Args:
            value (int): alignment size that will be stored.
        """
        assert alignment_size in [8, 16]
        log2_latent_downscale_factor, log2_z_downscale_factor = 4, 6
        self.log2_latent_downscale_factor = log2_latent_downscale_factor
        self.log2_z_downscale_factor = log2_z_downscale_factor
        self.log2_psi_downscale_factor = 5
        s = self.owner.owner.get_processed_img_shape()
        if alignment_size == 8:
            s = torch.Size([s[0] * 2, s[1] * 2])
        self.latent_downscale_factor = 2 ** (log2_latent_downscale_factor)
        self.z_downscale_factor = 2 ** (log2_z_downscale_factor)
        self.latent_shape = y_shape
        self.z_shape = torch.Size((y_shape[0], y_shape[1], math.ceil(s[0] / self.z_downscale_factor),
                                   math.ceil(s[1] / self.z_downscale_factor)))

        self.get_base_tile_params(self)
        self.z_extend = self.HyperDecoderOverlap
        #self.psi_overlap_in_latent_samples = self.McmOverlap >> 2

        self.set_enable(self.numHorRegions > 1 or self.numVerRegions > 1)

        if not self.is_enabled():
            self.numHorRegions, self.numVerRegions = 1, 1

        self.chroma = 1 if alignment_size == 8 else 0
        self.calculate_region_coordinates(s[0], s[1], self.chroma)
        self.img_tiles = self._init_latent_tiles(True, 0 if self.region_residual_in_its_own_substream_flag else 2 ** (log2_z_downscale_factor - self.chroma) * self.z_extend)
        self.img_tiles_withoutExtend = self._init_latent_tiles(True, 0)
        self.calculate_region_coordinates(s[0], s[1], log2_latent_downscale_factor)
        self.latent_tiles = self._init_latent_tiles()
        self.latent_tiles_withExtend = self._init_latent_tiles(True, 0 if self.region_residual_in_its_own_substream_flag else self.McmOverlap >> 1)
        self.calculate_region_coordinates(s[0], s[1], log2_z_downscale_factor)
        self.z_tiles = self._init_latent_tiles(True, 0 if self.region_residual_in_its_own_substream_flag else self.z_extend)

        self.z_tiles_withoutExtend = self._init_latent_tiles(True, 0)

        self.calculate_region_coordinates(s[0], s[1], self.log2_psi_downscale_factor)
        self.psi_tiles = self._init_latent_tiles()
        self.psi_tiles_withExtend = self._init_latent_tiles(True, 0 if self.region_residual_in_its_own_substream_flag else self.McmOverlap >> 2)

    def regions_generator(self):
        for latent_tile, latent_tile_woExtend, z_tile, z_tile_woExtend, img_tile, img_tile_woExtend in zip(
                self.latent_tiles_withExtend,
                self.latent_tiles,
                self.z_tiles,
                self.z_tiles_withoutExtend,
                self.img_tiles,
                self.img_tiles_withoutExtend):
            toDecode = True
            toEncode = True

            yield latent_tile, z_tile, img_tile, latent_tile_woExtend, z_tile_woExtend, img_tile_woExtend, toDecode, toEncode

    def psi_regions_generator(self):
        for psi_tile, psi_tile_wo, latent_tile, latent_tile_woExtend, z_tile, z_tile_woExtend, img_tile, img_tile_woExtend in zip(
                self.psi_tiles_withExtend,
                self.psi_tiles,
                self.latent_tiles_withExtend,
                self.latent_tiles,
                self.z_tiles,
                self.z_tiles_withoutExtend,
                self.img_tiles,
                self.img_tiles_withoutExtend):
            toDecode = True
            toEncode = True
      
            yield psi_tile, psi_tile_wo, latent_tile, z_tile, img_tile, latent_tile_woExtend, z_tile_woExtend, img_tile_woExtend, toDecode, toEncode

    def encode_header(self, ec: HeaderCoder) -> None:
        """Function for encoding header fields for tiling. I.e. tile size and overlap.

        Args:
            ec (HeaderCoder): entropy coding module that is used for encoding the data.
        """
        pass

    def decode_header(self, ec: HeaderCoder) -> None:
        """Function for decoding header fields for tiling. I.e. tile size and overlap.

        Args:
            ec (HeaderCoder): entropy coding module that is used for decoding the data.
        """
        pass

    def _init_latent_tiles(self, extend=False, extend_amount=0) -> TileGrid:
        """Split image into tiles of given width and height. All tiles except last row and last
        column need to have width and height, which are a multiple of `alignment_size`.
        Tiles are returned via a instance of `TileGrid`.

        Returns:
            TileGrid: The image partitioned into tiles. No actual data, just locations are stored
            in this.
        """

        latent_tiles = []
        ver = 0
        for tile_start_y, tile_end_y in zip(self.vertical_reg_coords_start, self.vertical_reg_coords_end):
            image_tile_row = []
            hor = 0
            for tile_start_x, tile_end_x in zip(self.horizontal_reg_coords_start, self.horizontal_reg_coords_end):
                x = tile_start_x
                y = tile_start_y
                height = tile_end_y - tile_start_y
                width = tile_end_x - tile_start_x

                if extend:
                    height += 2 * extend_amount
                    width += 2 * extend_amount

                    if hor == 0:
                        width -= extend_amount
                    else:
                        x -= extend_amount
                    if hor == len(self.horizontal_reg_coords_start) - 1:
                        width -= extend_amount

                    if ver == 0:
                        height -= extend_amount
                    else:
                        y -= extend_amount
                    if ver == len(self.vertical_reg_coords_start) - 1:
                        height -= extend_amount

                image_tile_row.append(
                    Area.from_x_y_width_height(x, y, width, height))
                hor += 1
            latent_tiles.append(image_tile_row)
            ver += 1
        rows = len(latent_tiles)
        columns = len(latent_tiles[0])

        return TileGrid(rows, columns, latent_tiles)

    def get_iter_over_tiles(self, input_data: torch.Tensor, output_data: torch.Tensor):
        for idx_y in range(self.latent_tiles.rows):
            for idx_x in range(self.latent_tiles.columns):
                yield TileIterObject(self, input_data, output_data, self.z_tiles,
                                     self.latent_tiles, self.is_enabled(), idx_x, idx_y)

    def __len__(self):
        return self.latent_tiles.rows * self.latent_tiles.columns

    def get_core_latent(self, latent_tile: Area, shift=None):
        if shift == None:
            shift = 2 ** (self.log2_z_downscale_factor - self.log2_latent_downscale_factor) * self.z_extend
        tile = Area.from_x_y_width_height(0 if latent_tile.position.x == 0 else shift,
                                          0 if latent_tile.position.y == 0 else shift, latent_tile.size.width,
                                          latent_tile.size.height)
        return tile

    def get_core_psi(self, latent_tile: Area, shift=None):
        if shift == None:
            shift = 2 ** (self.log2_psi_downscale_factor - self.log2_latent_downscale_factor) * self.z_extend
        tile = Area.from_x_y_width_height(0 if latent_tile.position.x == 0 else shift,
                                          0 if latent_tile.position.y == 0 else shift, latent_tile.size.width,
                                          latent_tile.size.height)
        return tile
