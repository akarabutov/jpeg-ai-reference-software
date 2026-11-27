"""Training data: random crops from a folder of images, with optional text
saliency masks (for TextFidelityLoss).

The paper trains on ~90k internal generic images + 2.3k text images + ~28k
images from DIV2K / CLIC / Flickr2K.  Any image folder(s) can be used here.

Text masks: the paper produces them with the CRAFT text detector [39].  We
expect them to be precomputed (see ``scripts/make_text_masks.py``) and stored as
single-channel PNGs with the same relative path (``.png`` extension) under
``mask_root``; images without a mask get an all-zero mask.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(roots: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for root in roots:
        root = Path(root)
        if root.is_file():
            files += [Path(l.strip()) for l in root.read_text().splitlines() if l.strip()]
        else:
            files += sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)
    return files


def load_rgb(path) -> torch.Tensor:
    """-> float tensor [3, H, W] in [0, 1]."""
    img = Image.open(path).convert("RGB")
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0


def save_rgb(x: torch.Tensor, path) -> None:
    arr = (x.clamp(0, 1) * 255.0 + 0.5).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path)


class CropDataset(Dataset):
    def __init__(
        self,
        roots: Sequence[str],
        crop_size: int = 256,
        mask_root: Optional[str] = None,
        random_downscale: bool = True,
        min_downscale: float = 0.5,
    ):
        self.files = list_images(roots)
        if not self.files:
            raise FileNotFoundError(f"no images found in {roots}")
        self.roots = [Path(r) for r in roots]
        self.crop = crop_size
        self.mask_root = Path(mask_root) if mask_root else None
        self.random_downscale = random_downscale
        self.min_downscale = min_downscale

    def __len__(self) -> int:
        return len(self.files)

    def _mask_path(self, path: Path) -> Optional[Path]:
        if self.mask_root is None:
            return None
        for root in self.roots:
            try:
                rel = path.relative_to(root)
                break
            except ValueError:
                continue
        else:
            rel = Path(path.name)
        cand = (self.mask_root / rel).with_suffix(".png")
        return cand if cand.exists() else None

    def __getitem__(self, idx: int):
        for _ in range(10):
            try:
                return self._load(self.files[idx])
            except (OSError, ValueError) as e:  # unreadable / truncated file
                print(f"warning: skipping {self.files[idx]}: {e}")
                idx = random.randrange(len(self.files))
        raise RuntimeError("too many unreadable images")

    def _load(self, path: Path):
        img = Image.open(path).convert("RGB")
        mpath = self._mask_path(path)
        mask = Image.open(mpath).convert("L") if mpath else None

        w, h = img.size
        # Random downscale of large images (reduces noise / JPEG artefacts of
        # the sources and diversifies scale); never below the crop size.
        if self.random_downscale and min(w, h) > 2 * self.crop:
            lo = max(self.min_downscale, self.crop / min(w, h))
            s = random.uniform(lo, 1.0)
            size = (max(self.crop, round(w * s)), max(self.crop, round(h * s)))
            img = img.resize(size, Image.BICUBIC)
            if mask is not None:
                mask = mask.resize(size, Image.NEAREST)
            w, h = img.size
        # Upscale images that are too small.
        if min(w, h) < self.crop:
            s = self.crop / min(w, h)
            size = (max(self.crop, round(w * s)), max(self.crop, round(h * s)))
            img = img.resize(size, Image.BICUBIC)
            if mask is not None:
                mask = mask.resize(size, Image.NEAREST)
            w, h = img.size

        x0 = random.randint(0, w - self.crop)
        y0 = random.randint(0, h - self.crop)
        box = (x0, y0, x0 + self.crop, y0 + self.crop)
        x = torch.from_numpy(np.asarray(img.crop(box)).copy()).permute(2, 0, 1).float() / 255.0
        if mask is not None:
            m = torch.from_numpy(np.asarray(mask.crop(box)).copy()).float()[None] / 255.0
        else:
            m = torch.zeros(1, self.crop, self.crop)
        if random.random() < 0.5:
            x, m = x.flip(-1), m.flip(-1)
        return {"x": x, "mask": (m > 0.5).float()}
