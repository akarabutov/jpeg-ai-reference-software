#!/usr/bin/env python3
"""Layer-wise post-training quantization of the PICO scale decoder.

Produces an integer-only scale decoder (INT8 weights, UINT8 activations,
integer requantization, GELU look-up tables, integer softplus/CDF-index
thresholds) so that the entropy-coding parameters are bit-exact on every
platform (Sec. 3.2 of arXiv:2605.05148).  See picocodec/quant.py.

Steps
  1. run the trained encoder on random 512x512 crops of the calibration images
     at several quality levels -> hyper-latent symbols k (and latent symbols y);
  2. calibrate layer by layer on the quantized prefix (weight scales, bias
     correction, activation ranges); print a per-layer report (SQNR);
  3. evaluate on --val images (default: the calibration crops): agreement of the
     CDF indexes with the float scale decoder and the real coding cost of the
     latent y with both;
  4. check determinism (threads / batch composition);
  5. save a copy of the checkpoint with the key "scale_decoder_int".  Codec
     tools (pico_codec.py, evaluate.py) use it automatically.

  python scripts/quantize_scale_decoder.py -m runs/perc/checkpoints/last.ckpt \
      --calib /data/calib_images --val /data/clic_valid --out runs/perc/pico_int.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from picocodec.data import list_images, load_rgb  # noqa: E402
from picocodec.layers import level_weights  # noqa: E402
from picocodec.lightning_module import extract_pico_state_dict  # noqa: E402
from picocodec.model import PICO, PICOConfig, default_scale_table  # noqa: E402
from picocodec.quant import QuantScaleDecoder, float_indexes  # noqa: E402


def random_crops(files, n, size, seed):
    rng = random.Random(seed)
    crops = []
    for i in range(n):
        x = load_rgb(files[i % len(files)])
        _, h, w = x.shape
        if h < size or w < size:  # replicate-pad small images
            x = F.pad(x[None], (0, max(0, size - w), 0, max(0, size - h)), mode="replicate")[0]
            _, h, w = x.shape
        t, l = rng.randint(0, h - size), rng.randint(0, w - size)
        crops.append(x[:, t:t + size, l:l + size])
    return torch.stack(crops)


@torch.no_grad()
def collect(model: PICO, x: torch.Tensor, levels, device, batch: int = 8):
    """-> hyper symbols k, latent symbols y^ (integers), per crop."""
    med = model.entropy_bottleneck._get_medians().detach().view(1, -1, 1, 1)
    ks, syms = [], []
    for i in range(0, len(x), batch):
        xb = x[i:i + batch].to(device)
        lvl = torch.tensor([levels[(i + j) % len(levels)] for j in range(len(xb))], device=device)
        lw = level_weights(lvl)
        y, z = model.encoder(xb, lw)
        k = torch.round(z - med.to(device))
        z_hat = k + med.to(device)
        p = model.context_decoder(z_hat)
        sym, _ = model.context.encode(y * model.gain(lw), p)
        ks.append(k.cpu().to(torch.int64))
        syms.append(sym.cpu())
    return torch.cat(ks), torch.cat(syms)


def coding_bits(model: PICO, sym: torch.Tensor, idx: torch.Tensor) -> float:
    """Ideal code length of y^ when coded with the scale-table entry `idx`
    (this is what the range coder actually uses)."""
    table = model.gaussian_conditional.scale_table.cpu().double()
    scales = table[idx.long()]
    lik = model.gaussian_conditional._likelihood(sym.double(), scales)
    return (-torch.log2(lik.clamp_min(1e-12))).sum().item()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", required=True, help="trained checkpoint (.ckpt or .pt)")
    ap.add_argument("--calib", required=True, help="folder with calibration images")
    ap.add_argument("--val", help="folder with validation images (default: calibration crops)")
    ap.add_argument("--num-crops", type=int, default=64, help="512x512 calibration crops")
    ap.add_argument("--num-val-crops", type=int, default=32)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--levels", type=float, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--percentile", type=float, default=0.9999,
                    help="activation range percentile (1.0 = min/max)")
    ap.add_argument("--no-bias-correction", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    model = PICO(PICOConfig.from_dict(ck.get("model_cfg")))
    model.load_state_dict(extract_pico_state_dict(ck))
    table = default_scale_table(model.cfg.scale_bound)
    model.update(scale_table=table, force=True)
    model.eval().to(args.device)
    model.scale_decoder.cpu()

    # ---- 1. calibration data ------------------------------------------------ #
    files = list_images([args.calib])
    print(f"calibration: {args.num_crops} crops {args.crop}x{args.crop} from {len(files)} images, "
          f"levels {args.levels}")
    k_cal, _ = collect(model, random_crops(files, args.num_crops, args.crop, args.seed),
                       args.levels, args.device)
    med = model.entropy_bottleneck._get_medians().detach().view(-1).cpu()

    # ---- 2. layer-wise calibration ------------------------------------------ #
    qsd, report = QuantScaleDecoder.calibrate(
        model.scale_decoder, k_cal, med, table,
        percentile=args.percentile, bias_correction=not args.no_bias_correction)
    print(f"\n{'layer':<16}{'shape':>18}{'scale':>12}{'zp':>6}{'SQNR, dB':>11}")
    for r in report:
        sc = f"{r['scale']:.3e}" if r["scale"] is not None else "-"
        zp = str(r["zp"]) if r["zp"] is not None else "-"
        print(f"{r['layer']:<16}{str(r['shape']):>18}{sc:>12}{zp:>6}{r['sqnr_db']:>11.2f}")
    print(f"int8 weights: {qsd.num_bytes() / 1e6:.2f} MB")

    # ---- 3. evaluation -------------------------------------------------------- #
    vfiles = list_images([args.val]) if args.val else files
    k_val, sym_val = collect(model, random_crops(vfiles, args.num_val_crops, args.crop, args.seed + 1),
                             args.levels, args.device)
    with torch.no_grad():
        sigma = model.scale_decoder(k_val.float() + med.view(1, -1, 1, 1))
    idx_f = float_indexes(sigma, table, model.cfg.scale_bound)
    idx_q = qsd(k_val)
    diff = (idx_q.long() - idx_f.long()).abs()
    bits_f = coding_bits(model, sym_val, idx_f)
    bits_q = coding_bits(model, sym_val, idx_q)
    pixels = args.num_val_crops * args.crop ** 2
    print(f"\nCDF index agreement with float scale decoder: {(diff == 0).float().mean() * 100:.2f}% "
          f"(|diff| = 1: {(diff == 1).sum().item()}, |diff| > 1: {(diff > 1).sum().item()} "
          f"of {diff.numel()} elements)")
    print(f"latent rate: float {bits_f / pixels:.5f} bpp, int {bits_q / pixels:.5f} bpp "
          f"({(bits_q / bits_f - 1) * 100:+.3f}%)")

    # ---- 4. determinism ------------------------------------------------------- #
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    single = torch.cat([qsd(k_val[i:i + 1]) for i in range(len(k_val))])
    torch.set_num_threads(max(2, threads))
    batched = qsd(k_val)
    torch.set_num_threads(threads)
    ok = torch.equal(single, batched) and torch.equal(batched, idx_q)
    print(f"determinism (1 thread / per-sample vs {max(2, threads)} threads / batched): "
          f"{'identical' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit("integer scale decoder is not deterministic")

    # ---- 5. save -------------------------------------------------------------- #
    ck["scale_decoder_int"] = qsd.to_dict()
    ck["scale_decoder_int_report"] = report
    torch.save(ck, args.out)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
