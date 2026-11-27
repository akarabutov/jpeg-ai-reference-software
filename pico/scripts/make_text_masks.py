#!/usr/bin/env python3
"""Precompute text saliency masks for TextFidelityLoss.

The paper uses the CRAFT text detector [39].  We use the CRAFT detector
shipped with EasyOCR (``pip install easyocr``).  For every image under
``--images`` a binary PNG mask (255 = text) is written to ``--out`` with the
same relative path.  Boxes are dilated by ``--dilate`` pixels.

  python scripts/make_text_masks.py --images /data/train --out /data/train_masks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from picocodec.data import list_images  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dilate", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=2560, help="detector input size limit")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    import easyocr
    reader = easyocr.Reader(["en"], gpu=args.gpu, recognizer=False)
    root, out = Path(args.images), Path(args.out)
    n_text = 0
    files = list_images([args.images])
    for i, f in enumerate(files):
        img = Image.open(f).convert("RGB")
        w, h = img.size
        s = min(1.0, args.max_side / max(w, h))
        small = img.resize((round(w * s), round(h * s))) if s < 1 else img
        horiz, free = reader.detect(np.asarray(small))
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        d = args.dilate
        for x0, x1, y0, y1 in horiz[0]:
            draw.rectangle([x0 / s - d, y0 / s - d, x1 / s + d, y1 / s + d], fill=255)
        for poly in free[0]:
            pts = [(px / s, py / s) for px, py in poly]
            draw.polygon(pts, fill=255)
        dst = (out / f.relative_to(root)).with_suffix(".png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        mask.save(dst)
        n_text += int(len(horiz[0]) + len(free[0]) > 0)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(files)} images, {n_text} with text", flush=True)
    print(f"done: {len(files)} masks, {n_text} images contain text")


if __name__ == "__main__":
    main()
