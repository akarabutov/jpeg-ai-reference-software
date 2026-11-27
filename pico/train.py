#!/usr/bin/env python3
"""Train PICO.

Two phases (Sec. 3.3 / 4.2 / App. C):
  1) ``configs/pico_mse.yaml``         -- MSE-only distortion,
     Adam lr 8e-4, decayed to 30% / 10% at 70% / 90% of training.
  2) ``configs/pico_perceptual.yaml``  -- MSE + LPIPS + MS-SSIM + TilingArtifactLoss
     + TextFidelityLoss + GAN, initialised from phase 1 (``--init``),
     lr decayed to 50% / 30% / 10% at 30% / 60% / 80% of training,
     GAN weight warmed up linearly.

A single model covers all bitrates: every sample gets a random fine quality
level l in {0, 0.1, ..., 7.0} and the loss uses alpha_l, lambda_l (App. F).

Examples:
  python train.py -c configs/pico_mse.yaml --out runs/mse \
      --set data.train=[/data/div2k,/data/flickr2k]
  python train.py -c configs/pico_perceptual.yaml --out runs/perc \
      --init runs/mse/last.pt --set data.train=[/data/div2k]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from picocodec.data import CropDataset, list_images, load_rgb
from picocodec.discriminator import PatchDiscriminator, discriminator_loss, generator_loss
from picocodec.layers import NUM_FINE_LEVELS, fine_index_to_level
from picocodec.losses import LossConfig, RateDistortionLoss, bits_per_pixel
from picocodec.model import PICO, PICOConfig, count_parameters


# --------------------------------------------------------------------------- #
# config helpers
# --------------------------------------------------------------------------- #
def apply_overrides(cfg: dict, overrides) -> dict:
    """--set a.b.c=value  (value parsed as YAML)."""
    for item in overrides or []:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def lr_lambda_factory(total: int, milestones, factors):
    def fn(step: int) -> float:
        frac = step / max(1, total)
        f = 1.0
        for m, fac in zip(milestones, factors):
            if frac >= m:
                f = fac
        return f
    return fn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(model: PICO, files, device, levels=(0.0, 3.5, 7.0), max_images: int = 24):
    """Estimated bpp (from likelihoods) and PSNR at a few levels; images are
    centre-cropped to at most 512x512 and to a multiple of 64."""
    model.eval()
    res = {}
    for lvl in levels:
        bpps, psnrs = [], []
        for f in files[:max_images]:
            x = load_rgb(f)
            h = min(512, x.shape[1] // 64 * 64)
            w = min(512, x.shape[2] // 64 * 64)
            if h == 0 or w == 0:
                continue
            t, l = (x.shape[1] - h) // 2, (x.shape[2] - w) // 2
            x = x[None, :, t:t + h, l:l + w].to(device)
            out = model(x, torch.tensor([lvl], device=device))
            bpps.append(bits_per_pixel(out["likelihoods"], h * w).item())
            mse = F.mse_loss(out["x_hat"].clamp(0, 1), x).item()
            psnrs.append(10 * math.log10(1 / max(mse, 1e-10)))
        if bpps:
            res[f"val/l{lvl}_bpp"] = float(np.mean(bpps))
            res[f"val/l{lvl}_psnr"] = float(np.mean(psnrs))
    model.train()
    return res


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--init", help="initialise model weights from a checkpoint (phase 1 -> 2)")
    ap.add_argument("--resume", help="resume training state from a checkpoint")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key.sub=value")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg = apply_overrides(cfg, args.set)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    tcfg, dcfg = cfg["train"], cfg["data"]
    seed_everything(tcfg.get("seed", 0))
    device = torch.device(args.device)
    phase = cfg["loss"].get("phase", "mse")

    # ---- model ----------------------------------------------------------- #
    model_cfg = PICOConfig.from_dict(cfg.get("model"))
    model = PICO(model_cfg).to(device)
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"initialised from {args.init}")
    print(f"params: encoder {count_parameters(model.encoder)/1e6:.2f}M, "
          f"decoder {count_parameters(model.decoder)/1e6:.2f}M, "
          f"total {count_parameters(model)/1e6:.2f}M")

    aux_params = [model.entropy_bottleneck.quantiles]
    main_params = [p for n, p in model.named_parameters() if not n.endswith(".quantiles")]
    opt = torch.optim.Adam(main_params, lr=tcfg["lr"])
    aux_opt = torch.optim.Adam(aux_params, lr=tcfg.get("aux_lr", 1e-3))
    total = int(tcfg["steps"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda_factory(total, tcfg["lr_milestones"], tcfg["lr_factors"]))

    criterion = RateDistortionLoss(LossConfig.from_dict(cfg["loss"])).to(device)

    disc = d_opt = None
    if phase == "perceptual" and criterion.cfg.w_gan > 0:
        disc = PatchDiscriminator(**cfg.get("disc", {}).get("arch", {})).to(device)
        dc = cfg.get("disc", {})
        d_opt = torch.optim.Adam(disc.parameters(), lr=dc.get("lr", 1e-4),
                                 betas=tuple(dc.get("betas", (0.5, 0.999))))
        print(f"discriminator: {count_parameters(disc)/1e6:.2f}M params")

    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        aux_opt.load_state_dict(ck["aux_opt"])
        sched.load_state_dict(ck["sched"])
        if disc is not None and "disc" in ck:
            disc.load_state_dict(ck["disc"])
            d_opt.load_state_dict(ck["d_opt"])
        step = ck["step"]
        print(f"resumed from {args.resume} @ step {step}")

    # ---- data ------------------------------------------------------------ #
    train_roots = dcfg["train"] if isinstance(dcfg["train"], list) else [dcfg["train"]]
    ds = CropDataset(train_roots, dcfg.get("crop_size", 256), dcfg.get("mask_root"),
                     dcfg.get("random_downscale", True))
    dl = DataLoader(ds, batch_size=dcfg.get("batch_size", 8), shuffle=True, drop_last=True,
                    num_workers=dcfg.get("num_workers", 4), pin_memory=device.type == "cuda",
                    persistent_workers=dcfg.get("num_workers", 4) > 0)
    val_files = list_images([dcfg["val"]]) if dcfg.get("val") else []
    use_text = phase == "perceptual" and bool(dcfg.get("mask_root"))
    print(f"train images: {len(ds)}, val images: {len(val_files)}")

    amp = bool(tcfg.get("amp", False)) and device.type == "cuda"
    grad_clip = tcfg.get("grad_clip", 1.0)
    log_every, ckpt_every = tcfg.get("log_every", 100), tcfg.get("ckpt_every", 10_000)
    val_every = tcfg.get("val_every", 10_000)
    log_file = open(out_dir / "log.jsonl", "a")

    def save(name: str) -> None:
        ck = {"model": model.state_dict(), "model_cfg": model_cfg.to_dict(),
              "opt": opt.state_dict(), "aux_opt": aux_opt.state_dict(),
              "sched": sched.state_dict(), "step": step, "phase": phase, "config": cfg}
        if disc is not None:
            ck["disc"], ck["d_opt"] = disc.state_dict(), d_opt.state_dict()
        torch.save(ck, out_dir / name)

    model.train()
    acc = defaultdict(float)
    n_acc = 0
    t0 = time.time()
    while step < total:
        for batch in dl:
            if step >= total:
                break
            x = batch["x"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True) if use_text else None
            b = x.shape[0]
            level = fine_index_to_level(
                torch.randint(0, NUM_FINE_LEVELS, (b,), device=device).float())

            # ---- codec (generator) step --------------------------------- #
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(x, level)
                g_loss = None
                if disc is not None:
                    disc.requires_grad_(False)
                    g_loss = generator_loss(disc(out["x_hat"]).float(), mask)
            out["x_hat"] = out["x_hat"].float()
            logs = criterion(out, x, level, mask, g_loss, step)

            opt.zero_grad(set_to_none=True)
            logs["loss"].backward()
            if grad_clip:
                logs["grad_norm"] = torch.nn.utils.clip_grad_norm_(main_params, grad_clip)
            opt.step()
            sched.step()

            aux_loss = model.aux_loss()
            aux_opt.zero_grad(set_to_none=True)
            aux_loss.backward()
            aux_opt.step()
            logs["aux"] = aux_loss.detach()

            # ---- discriminator step ------------------------------------ #
            if disc is not None:
                disc.requires_grad_(True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    d_loss = discriminator_loss(disc(x).float(), disc(out["x_hat"].detach()).float())
                d_opt.zero_grad(set_to_none=True)
                d_loss.backward()
                d_opt.step()
                logs["d_loss"] = d_loss.detach()

            step += 1
            for k, v in logs.items():
                acc[k] += float(v)
            n_acc += 1

            if step % log_every == 0 or step == total:
                rec = {k: v / n_acc for k, v in acc.items()}
                rec.update(step=step, lr=sched.get_last_lr()[0],
                           it_per_s=n_acc / (time.time() - t0))
                print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in rec.items()), flush=True)
                log_file.write(json.dumps(rec) + "\n")
                log_file.flush()
                acc.clear()
                n_acc = 0
                t0 = time.time()
            if val_files and (step % val_every == 0 or step == total):
                rec = validate(model, val_files, device)
                rec["step"] = step
                print("VAL", json.dumps(rec), flush=True)
                log_file.write(json.dumps(rec) + "\n")
            if step % ckpt_every == 0:
                save(f"step_{step:08d}.pt")
                save("last.pt")
    save("last.pt")
    print(f"done: {out_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
