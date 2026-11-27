#!/usr/bin/env python3
"""Evaluate a PICO checkpoint on a folder of images with real bitstreams.

For every image and quality level: encode -> bytes -> decode, then report the
actual bpp (file size), PSNR, MS-SSIM and (optionally) LPIPS.  Writes a CSV
with per-image rows and prints per-level averages.

  python evaluate.py -m runs/perc/last.pt -d /data/clic2020_test \
      -q 0 1 2 3 4 5 6 7 --lpips --out results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from pico_codec import load_model
from picocodec.codec import decode_image, encode_image
from picocodec.data import list_images, load_rgb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("-d", "--data", required=True)
    ap.add_argument("-q", "--qualities", type=float, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from pytorch_msssim import ms_ssim
    lp = None
    if args.lpips:
        import lpips
        lp = lpips.LPIPS(net="vgg", verbose=False).to(args.device).eval()

    model = load_model(args.model, args.device)
    files = list_images([args.data])
    rows = []
    agg = defaultdict(lambda: defaultdict(list))
    for f in files:
        x = load_rgb(f)
        for q in args.qualities:
            t0 = time.time()
            data = encode_image(model, x, q, device=args.device)
            t1 = time.time()
            x_hat = decode_image(model, data, device=args.device)
            t2 = time.time()
            xr = (x_hat * 255).round() / 255  # evaluate the 8-bit output
            mse = F.mse_loss(xr, x).item()
            row = {
                "image": str(f), "quality": q,
                "bpp": len(data) * 8 / (x.shape[1] * x.shape[2]),
                "psnr": 10 * math.log10(1 / max(mse, 1e-12)),
                "enc_s": t1 - t0, "dec_s": t2 - t1,
            }
            if min(x.shape[1:]) > 160:
                row["ms_ssim"] = ms_ssim(xr[None], x[None], data_range=1.0).item()
            if lp is not None:
                with torch.no_grad():
                    row["lpips"] = lp(xr[None].to(args.device) * 2 - 1,
                                      x[None].to(args.device) * 2 - 1).item()
            rows.append(row)
            for k, v in row.items():
                if k not in ("image", "quality"):
                    agg[q][k].append(v)
            print(", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in row.items()), flush=True)

    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "image", k != "quality", k))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("\nper-level averages:")
    for q in args.qualities:
        print(f"q={q}: " + ", ".join(f"{k}={sum(v)/len(v):.4f}" for k, v in agg[q].items()))


if __name__ == "__main__":
    main()
