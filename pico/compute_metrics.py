#!/usr/bin/env python3
"""Compute the training-loss metrics of PICO for every PNG in a dataset.

For each image (and each quality level): real encode -> bitstream -> decode
(8-bit output), then every term of the training loss (Sec. 4.2, Eq. 1) is
evaluated with the same functions that are used for training:

    D = MSE + w1 LPIPS + w2 (1 - MS-SSIM) + w3 TilingArtifactLoss
        + w4 TextFidelityLoss(m) + w5 GAN(m)
    rd_loss = alpha_l * (D + lambda_l * bpp)

One CSV row per image (per quality level):

    file, [quality,] bpp, mse, psnr, lpips, ms_ssim, msssim_loss, tal,
    text_l1, gan_g, d_logit, distortion, rd_loss

* bpp          -- real bitstream size in bits / pixels
* mse, psnr    -- on [0, 1] images
* lpips        -- LPIPS-VGG (as in the loss); --no-lpips to skip
* ms_ssim      -- MS-SSIM value; msssim_loss = 1 - MS-SSIM (loss term)
* tal          -- TilingArtifactLoss (multi-resolution L1 of low frequencies)
* text_l1      -- TextFidelityLoss; needs --masks (same relative path, .png)
* gan_g        -- generator (codec) adversarial loss with the checkpoint's
                  discriminator (Lightning checkpoints of phase 2), masked by
                  the text mask; d_logit = mean discriminator logit of x^
* distortion, rd_loss -- weighted sums with the weights / lambda_l / alpha_l of
                  the checkpoint's loss config (or --loss-config); terms that
                  are not available are left out of the sums.

Empty cells mean "not available" (e.g. no mask, no discriminator, image too
small for MS-SSIM).

  python compute_metrics.py -m runs/perc/checkpoints/last.ckpt -d /data/kodak \
      -q 3.5 --out kodak_metrics.csv
  python compute_metrics.py -m ... -d /data/clic -q 0 2 4 6 --masks /data/clic_masks
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from pico_codec import load_model
from picocodec.codec import decode_image, encode_image
from picocodec.discriminator import PatchDiscriminator, generator_loss
from picocodec.layers import fine_index_to_level, level_to_fine_index, level_weights, interp_table
from picocodec.losses import LossConfig, PerceptualMetrics, text_fidelity_loss, tiling_artifact_loss

METRICS = ["bpp", "mse", "psnr", "lpips", "ms_ssim", "msssim_loss", "tal", "text_l1",
           "gan_g", "d_logit", "distortion", "rd_loss"]


def list_pngs(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() == ".png")


def load_png(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0


def load_mask(mask_root: Optional[Path], rel: Path, shape) -> Optional[torch.Tensor]:
    if mask_root is None:
        return None
    p = (mask_root / rel).with_suffix(".png")
    if not p.exists():
        return torch.zeros(1, 1, *shape)  # no mask file = no text
    m = Image.open(p).convert("L").resize((shape[1], shape[0]), Image.NEAREST)
    return (torch.from_numpy(np.asarray(m).copy()).float()[None, None] > 127).float()


def training_config(ck: dict) -> dict:
    return (ck.get("hyper_parameters", {}) or {}).get("config") or ck.get("config") or {}


def load_discriminator(ck: dict, cfg: dict, device) -> Optional[PatchDiscriminator]:
    sd = ck.get("state_dict", {})
    dsd = {k[len("disc."):]: v for k, v in sd.items() if k.startswith("disc.")}
    if not dsd:
        return None
    disc = PatchDiscriminator(**(cfg.get("disc", {}) or {}).get("arch", {}))
    disc.load_state_dict(dsd)
    return disc.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", required=True, help="checkpoint (.ckpt / .pt)")
    ap.add_argument("-d", "--data", required=True, help="folder with PNG files (searched recursively)")
    ap.add_argument("-q", "--qualities", type=float, nargs="+", default=[3.5],
                    help="quality level(s) in [0, 7]; several -> a 'quality' column is added")
    ap.add_argument("--out", default="metrics.csv")
    ap.add_argument("--masks", help="folder with text masks (same relative paths, .png)")
    ap.add_argument("--loss-config", help="YAML with a 'loss' section to override the weights")
    ap.add_argument("--no-lpips", action="store_true", help="skip LPIPS (downloads VGG weights)")
    ap.add_argument("--no-gan", action="store_true", help="skip the discriminator metrics")
    ap.add_argument("--no-8bit", action="store_true",
                    help="evaluate the float reconstruction instead of the 8-bit output")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    # the codec works on a 0.1 grid of quality levels
    args.qualities = list(dict.fromkeys(
        fine_index_to_level(level_to_fine_index(q)) for q in args.qualities))
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    cfg = training_config(ck)
    loss_dict = dict(cfg.get("loss", {}) or {})
    if args.loss_config:
        loss_dict.update(yaml.safe_load(Path(args.loss_config).read_text()).get("loss", {}))
    lcfg = LossConfig.from_dict(loss_dict)
    lambdas = torch.tensor(lcfg.rate_lambdas, dtype=torch.float64)
    alphas = torch.tensor(lcfg.alphas, dtype=torch.float64)
    perceptual = lcfg.phase == "perceptual"

    model = load_model(args.model, args.device)
    disc = None if args.no_gan else load_discriminator(ck, cfg, device)
    metrics = PerceptualMetrics(lcfg.lpips_net)
    root = Path(args.data)
    mask_root = Path(args.masks) if args.masks else None
    files = list_pngs(root)
    if not files:
        raise SystemExit(f"no PNG files in {root}")
    multi_q = len(args.qualities) > 1
    print(f"{len(files)} images, qualities {args.qualities}, phase '{lcfg.phase}' weights, "
          f"LPIPS {'off' if args.no_lpips else 'on'}, "
          f"discriminator {'yes' if disc is not None else 'no'}, "
          f"masks {'yes' if mask_root else 'no'}")

    header = ["file"] + (["quality"] if multi_q else []) + METRICS
    sums: Dict[float, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    t_start = time.time()
    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for n, path in enumerate(files, 1):
            rel = path.relative_to(root)
            x_cpu = load_png(path)
            _, h, w = x_cpu.shape
            x = x_cpu[None].to(device)
            mask = load_mask(mask_root, rel, (h, w))
            mask = mask.to(device) if mask is not None else None
            for q in args.qualities:
                data = encode_image(model, x_cpu, q, device=args.device)
                x_hat = decode_image(model, data, device=args.device)
                if not args.no_8bit:
                    x_hat = torch.round(x_hat * 255) / 255
                x_hat = x_hat[None].to(device)
                r: Dict[str, Optional[float]] = {k: None for k in METRICS}
                with torch.no_grad():
                    r["bpp"] = len(data) * 8 / (h * w)
                    mse = F.mse_loss(x_hat, x).item()
                    r["mse"] = mse
                    r["psnr"] = 10 * math.log10(1 / max(mse, 1e-12))
                    if not args.no_lpips:
                        r["lpips"] = metrics.lpips(x, x_hat).item()
                    if min(h, w) > 32:
                        r["msssim_loss"] = metrics.ms_ssim_loss(x, x_hat).item()
                        r["ms_ssim"] = 1 - r["msssim_loss"]
                    r["tal"] = tiling_artifact_loss(x, x_hat, lcfg.tal_factors).item()
                    if mask is not None:
                        r["text_l1"] = text_fidelity_loss(x, x_hat, mask).item()
                    if disc is not None and min(h, w) >= 64:
                        logits = disc(x_hat)
                        r["gan_g"] = generator_loss(logits, mask, (cfg.get("disc", {}) or {})
                                                    .get("loss", "ns")).item()
                        r["d_logit"] = logits.mean().item()

                    # weighted distortion / RD loss as in training (available terms only)
                    dist = r["mse"]
                    if perceptual:
                        for key, wgt in (("lpips", lcfg.w_lpips), ("msssim_loss", lcfg.w_msssim),
                                         ("tal", lcfg.w_tiling), ("text_l1", lcfg.w_text),
                                         ("gan_g", lcfg.w_gan)):
                            if r[key] is not None:
                                dist += wgt * r[key]
                    lw = level_weights(torch.tensor([q])).double()
                    lam = interp_table(lambdas, lw).item()
                    alpha = interp_table(alphas, lw).item()
                    r["distortion"] = dist
                    r["rd_loss"] = alpha * (dist + lam * r["bpp"])

                row = [str(rel)] + ([q] if multi_q else [])
                row += ["" if r[k] is None else f"{r[k]:.6g}" for k in METRICS]
                writer.writerow(row)
                fh.flush()
                for k, v in r.items():
                    if v is not None:
                        sums[q][k].append(v)
            print(f"[{n}/{len(files)}] {rel}  "
                  + "  ".join(f"q={q:g}: {sums[q]['bpp'][-1]:.4f} bpp, {sums[q]['psnr'][-1]:.2f} dB"
                              for q in args.qualities), flush=True)

    print(f"\nwrote {args.out} ({time.time() - t_start:.1f}s). Averages:")
    for q in args.qualities:
        print(f"q={q:g}: " + ", ".join(f"{k}={sum(v) / len(v):.5g}" for k, v in sums[q].items()))


if __name__ == "__main__":
    main()
