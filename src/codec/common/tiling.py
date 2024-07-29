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
from typing import List, Generator

import torch
from attrs import define, field, validators

from src.codec.common import Image


@define(frozen=True, order=True)
class Position:
    """Data class for positions. Position has attributes x and y.
    """
    x: int = field(validator=validators.instance_of(int))
    y: int = field(validator=validators.instance_of(int))


@define(frozen=True, order=True)
class Size:
    """Data class for sizes of tiles. Size has attributes width and height.
    """
    width: int = field(validator=validators.instance_of(int))
    height: int = field(validator=validators.instance_of(int))


@define(frozen=True, order=True)
class Area:
    """Data class for defining areas in an image. Area has attributes position and size.
    """
    position: Position = field(validator=validators.instance_of(Position))
    size: Size = field(validator=validators.instance_of(Size))

    def __repr__(self) -> str:
        return f'Area({self.position}, {self.size})'

    @classmethod
    def from_position_size(cls, position: Position, size: Size) -> 'Area':
        """Initialize an instance of `Area` from a given position and size.

        Args:
            position (Position): position of the new `Area`.
            size (Size): size of the new `Area`.

        Returns:
            Area: the new instance of `Area`
        """
        return cls(position, size)

    @classmethod
    def from_x_y_width_height(cls, x: int, y: int, width: int, height: int) -> 'Area':
        """Initialize an instance of `Area` from a x and y location and given width and height.

        Args:
            x (int): x location of the new `Area`
            y (int): y location of the new `Area`
            width (int): width of the new `Area`
            height (int): height of the new `Area`

        Returns:
            Area: the new instance of `Area`
        """
        return cls(Position(x, y), Size(width, height))

    @classmethod
    def from_x_y_size(cls, x: int, y: int, size: Size) -> 'Area':
        """Initialize an instance of `Area` from a x and y location and given size.

        Args:
            x (int): x location of the new `Area`
            y (int): y location of the new `Area`
            size (Size): size of the new `Area`.

        Returns:
            Area: the new instance of `Area`
        """
        return cls(Position(x, y), size)

    @classmethod
    def from_position_width_height(cls, position: Position, width: int, height: int) -> 'Area':
        """Initialize an instance of `Area` from a given position and given width and height.

        Args:
            position (Position): position of the new `Area`.
            width (int): width of the new `Area`
            height (int): height of the new `Area`

        Returns:
            Area: the new instance of `Area`
        """
        return cls(position, Size(width, height))

    def top_left(self) -> Position:
        """Get postion of the top-left corner of this area.

        Returns:
            Position: postion of the top-left corner of this area.
        """
        return self.position

    def top_right(self) -> Position:
        """Get postion of the top-right corner of this area.

        Returns:
            Position: postion of the top-right corner of this area.
        """
        return Position(self.position.x + self.size.width - 1, self.position.y)

    def bottom_left(self) -> Position:
        """Get postion of the bottom-left corner of this area.

        Returns:
            Position: postion of the bottom-left corner of this area.
        """
        return Position(self.position.x, self.position.y + self.size.height - 1)

    def bottom_right(self) -> Position:
        """Get postion of the bottom-right corner of this area.

        Returns:
            Position: postion of the bottom-right corner of this area.
        """
        return Position(self.position.x + self.size.width - 1,
                        self.position.y + self.size.height - 1)

    def upscale(self, factor: int) -> 'Area':
        """Get a new Area scaled up by the integer `factor`.

        Args:
            factor (int): Ratio to scale by.

        Returns:
            Area: new Area scaled up by the integer `factor`
        """
        new_position = Position(self.position.x * factor, self.position.y * factor)
        new_size = Size(self.size.width * factor, self.size.height * factor)

        new_tile = Area.from_position_size(new_position, new_size)
        # new_tile.upscale_(factor)
        return new_tile

    def downscale(self, factor: int) -> 'Area':
        """Get a new Area scaled down by the integer `factor`.
        Note that rounding can occur here.

        Args:
            factor (int): Ratio to scale by.

        Returns:
            Area: new Area scaled down by the integer `factor`
        """
        new_position = Position(math.ceil(self.position.x / factor),
                                 math.ceil(self.position.y / factor))
        new_size = Size(math.ceil(self.size.width / factor), math.ceil(self.size.height / factor))

        new_tile = Area.from_position_size(new_position, new_size)
        # new_tile.downscale_(factor)
        return new_tile

    def contains(self, point: Position):
        
        x_is_inside =  self.top_left().x <= point.x <= self.bottom_right().x
        y_is_inside =  self.top_left().y <= point.y <= self.bottom_right().y

        return x_is_inside and y_is_inside

    def overlaps_with(self, other: 'Area'):
        overlaps = False

        corners = [ self.top_left(), 
                    self.top_right(), 
                    self.bottom_left(),
                    self.bottom_right()]

        for corner in corners:
            inside_other = other.contains(corner)
            overlaps = overlaps or inside_other

        return overlaps

    # def upscale_(self, factor: int) -> None:
    #     """Scale this area up by the integer `factor`.

    #     Args:
    #         factor (int): Ratio to scale by.
    #     """
    #     self.position = Position(self.position.x * factor, self.position.y * factor)
    #     self.size = Size(self.size.width * factor, self.size.height * factor)

    # def downscale_(self, factor: int) -> None:
    #     """Scale this area down by the integer `factor`.
    #     Note that rounding can occur here.

    #     Args:
    #         factor (int): Ratio to scale by.
    #     """
    #     self.position = Position(math.ceil(self.position.x / factor),
    #                              math.ceil(self.position.y / factor))
    #     self.size = Size(math.ceil(self.size.width / factor), math.ceil(self.size.height / factor))


@define
class TileGrid:
    """Data class to store information about tiles. Each tile is an instance of `tiling.Area`, which has
    members `tiling.Position` and `tiling.Size`.

    Yields:
        Area: A tile in this tile grid.
    """
    rows: int = field(validator=validators.instance_of(int))
    columns: int = field(validator=validators.instance_of(int))
    tiles: List[List[Area]] = field(
        repr=lambda tiles: '\n' + ''.join([str(row) + '\n' for row in tiles]))

    def get_column(self, idx: int) -> List[Area]:
        """Get all tiles in the column `idx` of this tile grid.

        Args:
            idx (int): index of the column of tiles to return.

        Returns:
            List[Area]: the column of tiles specified by `idx`
        """
        column = [row[idx] for row in self.tiles]
        return column

    def get_row(self, idx: int) -> List[Area]:
        """Get all tiles in the row `idx` of this tile grid.

        Args:
            idx (int): index of the row of tiles to return.

        Returns:
            List[Area]: the row of tiles specified by `idx`
        """
        row = self.tiles[idx]
        return row

    def get_num_tiles(self) -> int:
        """Get the number of tiles of this tile grid.

        Returns:
            int: number of tiles of this grid.
        """
        
        return self.rows * self.columns

    def get_tile(self, row_idx: int, column_idx: int) -> Area:
        """Get the tile in row `row_idx` and column `column_idx`.

        Args:
            row_idx (int): index of row to get tile from.
            column_idx (int): index of column to get tile from.

        Returns:
            Area: the tile in row `row_idx` and column `column_idx`.
        """
        return self.tiles[row_idx][column_idx]

    def __iter__(self):
        for tile_row in self.tiles:
            for tile in tile_row:
                yield tile

    def __reversed__(self):
        for tile_row in self.tiles[::-1]:
            for tile in tile_row[::-1]:
                yield tile

def get_alignment_size(num_downsampling_layers: int) -> int:
    """Compute alignment size for tiles from the number of downsampling layers in a neural network.

    Args:
        num_downsampling_layers (int): number of downsampling layers in a neural network

    Returns:
        int: alignment size for tiles
    """
    alignment_size = 2**(num_downsampling_layers + 1)
    return alignment_size


def get_data(image_or_features: torch.Tensor, tile: Area) -> torch.Tensor:
    """Retrieve the part of image or feature specified by a tile.

    Args:
        image_or_features (torch.Tensor): image or feature to retrieve the data from.
        tile (Area): specification of the area that should be taken.

    Returns:
        torch.Tensor: Retrieved part, which has size of `tile` (last two/spatial dimensions)
    """
    h,w = image_or_features.shape[-2:]
    if tile.size.height == h and tile.size.width == w:
        return image_or_features
    else:
        return image_or_features[..., tile.position.y:tile.position.y + tile.size.height,
                             tile.position.x:tile.position.x + tile.size.width]


def get_data_image_comp(image: Image, tile: Area, comp_name: str) -> torch.Tensor:
    """Retrieve the part of image or feature specified by a tile.

    Args:
        image (Image): image or feature to retrieve the data from.
        tile (Area): specification of the area that should be taken.
        comp_name (str): name of the compoennt

    Returns:
        torch.Tensor: Retrieved part, which has size of `tile` (last two/spatial dimensions)
    """

    return get_data(image.get_component(comp_name), tile)


def get_data_image(image: Image, tile: Area) -> Image:
    """Retrieve the part of image or feature specified by a tile.

    Args:
        image_or_features (Image): image or feature to retrieve the data from.
        tile (Area): specification of the area that should be taken.

    Returns:
        Image: Retrieved part, which has size of `tile` (last two/spatial dimensions)
    """

    tmp_comp = {}
    comp_name = Image.valid_comp_names[0]
    tmp_comp[comp_name] = get_data(image.get_component(comp_name), tile)
    if image.is_420():
        tile.downscale_(2)
    for c in Image.valid_comp_names[1:]:
        tmp_comp[c] = get_data(image.get_component(c), tile)
    return Image.create_from_tensors(tmp_comp['a'], tmp_comp['b'], tmp_comp['c'], image.data_range,
                                     image.bit_depth, image.color_space, image.format)


def assign_data(image_or_features: torch.Tensor, tile: Area, tile_data: torch.Tensor) -> torch.Tensor:
    """Assign the part of image or feature specified by a tile with the given data.

    Args:
        image_or_features (torch.Tensor):  image or feature to assigned the data to.
        tile (Area): specification of the area that should be assigned.
        tile_data (torch.Tensor): data that will be assigned.
    """
    h,w = image_or_features.shape[-2:]
    h = min(h - tile.position.y, tile_data.shape[-2])
    w = min(w - tile.position.x, tile_data.shape[-1])
    image_or_features[..., tile.position.y:tile.position.y + h,
                    tile.position.x:tile.position.x + w] = tile_data[..., :h, :w]
    return image_or_features


def assign_data2image(image: Image, tile: Area, tile_data: torch.Tensor) -> Image:
    """Assign data to a region of an `Image`.

    Args:
        image (Image): image consisting of 3 components. E.g.: Y,U,V or R,G,B.
        tile (Area): tile specifying the region in the image to which to assing datat to.
        tile_data (torch.Tensor): data that will be assigned to the image components.
    """
    comp_name = Image.valid_comp_names[0]
    t = image.get_component(comp_name)
    t = assign_data(t, tile, tile_data[:, 0:1])
    image.set_component(comp_name, t)
    if image.is_420():
        tile.upscale_(2)

    for i, comp_name in enumerate(Image.valid_comp_names[1:]):
        t = image.get_component(comp_name)
        t = assign_data(t, tile, tile_data[:, (i + 1):(i + 2)])
        image.set_component(comp_name, t)
    return image


@define(frozen=True)
class ColocatedTiles:
    """Data class for sizes of tiles. Size has attributes width and height.
    """
    img: Area = field(validator=validators.instance_of(Area))
    y: Area = field(validator=validators.instance_of(Area))
    psi: Area = field(validator=validators.instance_of(Area))
    z: Area = field(validator=validators.instance_of(Area))
    

    @staticmethod
    def iter_colocated_grids( tile_manager: "TileManager" ) -> Generator["ColocatedTiles", None, None]:
        """Generator method for interating over colocated TileGrids

        Returns:
            Generator[ColocatedTiles]: generator object
        """
        img_grid = tile_manager.image_tiles
        y_grid = tile_manager.latent_tiles
        psi_grid = tile_manager.latent_tiles_psi
        z_grid = tile_manager.latent_tiles_z

        tiles_descriptions = zip( img_grid, y_grid, psi_grid, z_grid)
        for img_tile, y_tile, psi_tile, z_tile in tiles_descriptions:
            yield ColocatedTiles(img_tile, y_tile, psi_tile, z_tile)
