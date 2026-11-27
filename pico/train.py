#!/usr/bin/env python3
"""Train PICO with PyTorch Lightning (TensorBoard logging).

Two phases (Sec. 3.3 / 4.2 / App. C):
  1) ``configs/pico_mse.yaml``         -- MSE-only distortion,
     Adam lr 8e-4, decayed to 30% / 10% at 70% / 90% of training.
  2) ``configs/pico_perceptual.yaml``  -- MSE + LPIPS + MS-SSIM + TilingArtifactLoss
     + TextFidelityLoss + GAN, initialised from phase 1 (``--init``),
     lr decayed to 50% / 30% / 10% at 30% / 60% / 80% of training.

Examples:
  python train.py -c configs/pico_mse.yaml --out runs/mse \
      --set "data.train=[/data/div2k,/data/flickr2k]" data.val=/data/clic_valid
  python train.py -c configs/pico_perceptual.yaml --out runs/perc \
      --init runs/mse/checkpoints/last.ckpt --set "data.train=[/data/div2k]"
  tensorboard --logdir runs

Multi-GPU:  --set train.devices=4 train.strategy=ddp_find_unused_parameters_true
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from picocodec.data import CenterCropDataset, CropDataset
from picocodec.lightning_module import PICOLightning
from picocodec.model import count_parameters


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


def build_loaders(cfg: dict):
    d = cfg["data"]
    roots = d["train"] if isinstance(d["train"], list) else [d["train"]]
    ds = CropDataset(roots, d.get("crop_size", 256), d.get("mask_root"),
                     d.get("random_downscale", True))
    nw = d.get("num_workers", 4)
    train = DataLoader(ds, batch_size=d.get("batch_size", 8), shuffle=True, drop_last=True,
                       num_workers=nw, pin_memory=torch.cuda.is_available(),
                       persistent_workers=nw > 0)
    val = None
    if d.get("val"):
        vds = CenterCropDataset([d["val"]], d.get("val_crop", 512),
                                max_images=d.get("val_max_images", 24))
        val = DataLoader(vds, batch_size=1, num_workers=min(nw, 2))
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--out", required=True, help="output directory (TensorBoard logs + checkpoints)")
    ap.add_argument("--init", help="initialise codec weights from a checkpoint (phase 1 -> 2)")
    ap.add_argument("--resume", help="resume the full training state from a Lightning .ckpt")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key.sub=value")
    args = ap.parse_args()

    cfg = apply_overrides(yaml.safe_load(Path(args.config).read_text()), args.set)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    tr = cfg["train"]
    L.seed_everything(tr.get("seed", 0), workers=True)

    precision = str(tr.get("precision", "32-true"))
    if precision.startswith("16"):
        raise ValueError("fp16 with a GradScaler is not supported (manual gradient clipping); "
                         "use precision=32-true or bf16-mixed")

    module = PICOLightning(cfg)
    if args.init:
        module.load_pico_weights(args.init)
        print(f"initialised codec from {args.init}")
    print(f"params: encoder {count_parameters(module.model.encoder)/1e6:.2f}M, "
          f"decoder {count_parameters(module.model.decoder)/1e6:.2f}M, "
          f"codec total {count_parameters(module.model)/1e6:.2f}M"
          + (f", discriminator {count_parameters(module.disc)/1e6:.2f}M" if module.disc else ""))

    train_dl, val_dl = build_loaders(cfg)
    k = module.opt_steps_per_iter  # Lightning counts optimizer steps, not iterations
    ckpt = ModelCheckpoint(
        dirpath=out / "checkpoints",
        filename="iter_{train/iteration:.0f}",
        auto_insert_metric_name=False,
        every_n_train_steps=int(tr.get("ckpt_every", 10_000)) * k,
        save_top_k=-1, save_last=True,
    )
    logger = TensorBoardLogger(save_dir=str(out), name="tb", default_hp_metric=False)
    trainer = L.Trainer(
        accelerator=tr.get("accelerator", "auto"),
        devices=tr.get("devices", 1),
        strategy=tr.get("strategy", "auto"),
        precision=precision,
        max_steps=int(tr["steps"]) * k,
        logger=logger,
        callbacks=[ckpt, TQDMProgressBar(refresh_rate=tr.get("progress_every", 20))],
        log_every_n_steps=int(tr.get("log_every", 100)),
        val_check_interval=int(tr.get("val_every", 10_000)) if val_dl is not None else None,
        check_val_every_n_epoch=None,
        limit_val_batches=1.0 if val_dl is not None else 0,
        num_sanity_val_steps=0,
        enable_model_summary=False,
        deterministic=False,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*does not have many workers.*")
        trainer.fit(module, train_dl, val_dl, ckpt_path=args.resume)
    print(f"done; checkpoints in {out / 'checkpoints'}, TensorBoard: tensorboard --logdir {out}")


if __name__ == "__main__":
    main()
