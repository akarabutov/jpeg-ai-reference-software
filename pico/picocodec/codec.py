"""Tiled encoding/decoding and the bitstream container (Sec. 3.2).

Each image is partitioned into non-overlapping 504x504 tiles.  For encoding,
every tile is extended by a 4-pixel contextual margin taken from neighbouring
tiles (replicate padding at the image border), giving 512x512 network inputs.
Edge tiles that are smaller than 504 px are processed at the smallest multiple
of 64 that fits.  The decoder reconstructs each tile independently and crops
the margin away.

Bitstream layout (big endian):
    magic "PICO" | version u8 | height u32 | width u32 | fine level u8 |
    for every tile in raster order: len(z) u32, z bytes, len(y) u32, y bytes
"""

from __future__ import annotations

import io
import math
import struct
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F

from .layers import fine_index_to_level, level_to_fine_index
from .model import PICO

MAGIC = b"PICO"
VERSION = 1
TILE = 504
MARGIN = 4
ALIGN = 64  # total downsampling factor of the hyper latent


@dataclass
class Tile:
    row: int
    col: int
    y0: int  # top-left of the tile content in image coordinates
    x0: int
    h: int   # content size (<= TILE)
    w: int
    ph: int  # processed (network input) size, multiple of ALIGN
    pw: int


def _proc_size(n: int) -> int:
    return int(math.ceil((n + 2 * MARGIN) / ALIGN) * ALIGN)


def tile_grid(height: int, width: int, tile: int = TILE) -> List[Tile]:
    tiles = []
    for r in range(math.ceil(height / tile)):
        for c in range(math.ceil(width / tile)):
            y0, x0 = r * tile, c * tile
            h, w = min(tile, height - y0), min(tile, width - x0)
            tiles.append(Tile(r, c, y0, x0, h, w, _proc_size(h), _proc_size(w)))
    return tiles


def extract_tile(x: torch.Tensor, t: Tile) -> torch.Tensor:
    """x: [1, 3, H, W] -> [1, 3, ph, pw]: tile + margin from neighbours,
    replicate-padded where it leaves the image."""
    _, _, H, W = x.shape
    top, left = t.y0 - MARGIN, t.x0 - MARGIN
    bottom, right = top + t.ph, left + t.pw
    crop = x[:, :, max(0, top):min(H, bottom), max(0, left):min(W, right)]
    pad = (max(0, -left), max(0, right - W), max(0, -top), max(0, bottom - H))
    if any(pad):
        crop = F.pad(crop, pad, mode="replicate")
    return crop


def _write_bytes(buf: io.BytesIO, b: bytes) -> None:
    buf.write(struct.pack(">I", len(b)))
    buf.write(b)


def _read_bytes(buf: io.BytesIO) -> bytes:
    (n,) = struct.unpack(">I", buf.read(4))
    return buf.read(n)


@torch.no_grad()
def encode_image(model: PICO, x: torch.Tensor, level: float, device=None) -> bytes:
    """x: [3, H, W] or [1, 3, H, W] in [0, 1]; level in [0, 7] (0.1 steps)."""
    if x.dim() == 3:
        x = x[None]
    device = device or next(model.encoder.parameters()).device
    fine = level_to_fine_index(level)
    lvl = fine_index_to_level(fine)
    _, _, H, W = x.shape
    buf = io.BytesIO()
    buf.write(MAGIC)
    buf.write(struct.pack(">BIIB", VERSION, H, W, fine))
    for t in tile_grid(H, W):
        out = model.compress(extract_tile(x, t).to(device), lvl)
        y_strings, z_strings = out["strings"]
        _write_bytes(buf, z_strings[0])
        _write_bytes(buf, y_strings[0])
    return buf.getvalue()


def read_header(data: bytes) -> Tuple[int, int, int]:
    if data[:4] != MAGIC:
        raise ValueError("not a PICO bitstream")
    version, H, W, fine = struct.unpack(">BIIB", data[4:14])
    if version != VERSION:
        raise ValueError(f"unsupported bitstream version {version}")
    return H, W, fine


@torch.no_grad()
def decode_image(model: PICO, data: bytes, device=None) -> torch.Tensor:
    """-> [3, H, W] in [0, 1]."""
    H, W, fine = read_header(data)
    lvl = fine_index_to_level(fine)
    device = device or next(model.decoder.parameters()).device
    buf = io.BytesIO(data[14:])
    out = torch.zeros(3, H, W)
    for t in tile_grid(H, W):
        z = _read_bytes(buf)
        y = _read_bytes(buf)
        shape = (t.ph // ALIGN, t.pw // ALIGN)
        rec = model.decompress([[y], [z]], shape, lvl, device=device)[0].cpu()
        out[:, t.y0:t.y0 + t.h, t.x0:t.x0 + t.w] = rec[:, MARGIN:MARGIN + t.h, MARGIN:MARGIN + t.w]
    return out
