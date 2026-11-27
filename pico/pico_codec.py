#!/usr/bin/env python3
"""PICO encoder / decoder command line.

  python pico_codec.py encode -m runs/perc/last.pt -q 3.5 input.png out.pico
  python pico_codec.py decode -m runs/perc/last.pt out.pico recon.png

The quality level -q is in [0, 7] with 0.1 granularity (71 levels, App. F);
0 is the lowest bitrate.  Neural transforms run on --device; the scale decoder
and entropy models always run on the CPU (Sec. 3.2).  If the checkpoint
contains an integer scale decoder (scripts/quantize_scale_decoder.py), it is
used, which makes the bitstream decodable bit-exactly on any platform.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from picocodec.codec import decode_image, encode_image, read_header
from picocodec.data import load_rgb, save_rgb
from picocodec.lightning_module import extract_pico_state_dict
from picocodec.model import PICO, PICOConfig
from picocodec.quant import QuantScaleDecoder


def load_model(path: str, device: str, reparameterize: bool = True,
               int_scale_decoder: bool = True) -> PICO:
    """Load a checkpoint (exported, Lightning .ckpt, or one produced by
    scripts/quantize_scale_decoder.py).  If it contains an integer scale
    decoder it is used unless `int_scale_decoder=False`."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = PICO(PICOConfig.from_dict(ck.get("model_cfg")))
    model.load_state_dict(extract_pico_state_dict(ck))
    model.update(force=True)
    if int_scale_decoder and "scale_decoder_int" in ck:
        model.set_int_scale_decoder(QuantScaleDecoder.from_dict(ck["scale_decoder_int"]))
    return model.to_inference(device, reparameterize=reparameterize)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("encode", "decode"):
        p = sub.add_parser(name)
        p.add_argument("-m", "--model", required=True)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--no-reparam", action="store_true",
                       help="keep ConvScale/Haar training parameterisation")
        p.add_argument("--float-scale-decoder", action="store_true",
                       help="ignore an integer scale decoder stored in the checkpoint")
        p.add_argument("input")
        p.add_argument("output")
        if name == "encode":
            p.add_argument("-q", "--quality", type=float, default=3.5)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = False
    model = load_model(args.model, args.device, not args.no_reparam,
                       int_scale_decoder=not args.float_scale_decoder)
    t0 = time.time()
    if args.cmd == "encode":
        x = load_rgb(args.input)
        data = encode_image(model, x, args.quality, device=args.device)
        Path(args.output).write_bytes(data)
        bpp = len(data) * 8 / (x.shape[1] * x.shape[2])
        print(f"{args.output}: {len(data)} bytes, {bpp:.4f} bpp, {time.time() - t0:.2f}s")
    else:
        data = Path(args.input).read_bytes()
        H, W, _, _ = read_header(data)
        x_hat = decode_image(model, data, device=args.device)
        save_rgb(x_hat, args.output)
        print(f"{args.output}: {W}x{H}, {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
