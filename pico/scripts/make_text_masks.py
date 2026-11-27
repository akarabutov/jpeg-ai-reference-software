#!/usr/bin/env python3
"""Precompute text saliency masks for TextFidelityLoss.

For every image under --images a binary PNG mask (255 = text) is written to
--out with the same relative path (.png).  Pipeline (see picocodec/textmask.py):

  1. text probability map at the ORIGINAL resolution, from overlapping tiles
     (--tile / --overlap), with CRAFT (the detector of the paper, weights from
     EasyOCR) or the PaddleOCR DB detector (--detector paddle);
  2. connected components of p > --low-thresh whose peak is >= --seed-thresh
     and area >= --min-area;
  3. optional (--recognizer easyocr|paddle): candidates with peak <
     --verify-below are read by a recognizer (also rotated by 90/180/270 deg)
     and kept only with confidence >= --min-conf and >= --min-chars
     alphanumeric characters.  OFF by default: on real photos the recognizer
     also rejects genuine text it cannot read (non-Latin scripts unless
     --langs covers them, handwriting, logos, multi-line regions); enable it
     only after checking --debug-dir / --report on your data;
  4. dilation by --dilate pixels.

Extras: --save-prob DIR writes the probability maps (8-bit PNG),
--debug-dir DIR writes overlays (kept mask red, rejected candidates blue),
--report CSV writes one row per candidate (box, peak, text, confidence, kept).

Examples:
  # CRAFT probability-map masks (default, no recognition filter)
  python scripts/make_text_masks.py --images /data/train --out /data/train_masks --gpu
  # PaddleOCR detector
  python scripts/make_text_masks.py --images /data/train --out /data/train_masks_pp \\
      --detector paddle
  # with the recognition filter for weak detections only, English + Korean
  python scripts/make_text_masks.py --images /data/train --out /data/m \\
      --recognizer easyocr --langs en ko --verify-below 0.8
  # check a few images visually
  python scripts/make_text_masks.py --images /data/sample --out /tmp/m --debug-dir /tmp/dbg
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from picocodec.data import list_images  # noqa: E402
from picocodec.textmask import (  # noqa: E402
    CraftDetector, EasyOCRRecognizer, MaskParams, PaddleDetector, PaddleRecognizer,
    build_mask, overlay, tiled_prob_map,
)


def build_models(args):
    reader = None

    def easy_reader():
        nonlocal reader
        if reader is None:
            import easyocr
            reader = easyocr.Reader(args.langs, gpu=args.gpu, verbose=False)
        return reader

    paddle_dev = ("gpu" if args.gpu else "cpu")
    if args.detector == "craft":
        det = CraftDetector(reader=easy_reader())
    else:
        det = PaddleDetector(model_name=args.paddle_det_model, device=paddle_dev)
    rec_name = args.recognizer
    if rec_name == "auto":
        rec_name = "easyocr" if args.detector == "craft" else "paddle"
    if rec_name == "none":
        rec = None
    elif rec_name == "easyocr":
        rec = EasyOCRRecognizer(reader=easy_reader())
    else:
        rec = PaddleRecognizer(model_name=args.paddle_rec_model, device=paddle_dev)
    return det, rec, rec_name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--detector", choices=["craft", "paddle"], default="craft")
    ap.add_argument("--recognizer", choices=["none", "easyocr", "paddle", "auto"], default="none",
                    help="recognition filter (off by default); auto = EasyOCR for craft, "
                         "PaddleOCR for paddle")
    ap.add_argument("--langs", nargs="+", default=["en"], help="EasyOCR languages")
    ap.add_argument("--paddle-det-model", default=None, help="e.g. PP-OCRv5_server_det")
    ap.add_argument("--paddle-rec-model", default=None, help="e.g. PP-OCRv5_server_rec")
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=128)
    p = MaskParams()
    ap.add_argument("--low-thresh", type=float, default=p.low_thresh)
    ap.add_argument("--seed-thresh", type=float, default=p.seed_thresh)
    ap.add_argument("--min-area", type=int, default=p.min_area)
    ap.add_argument("--min-conf", type=float, default=p.min_conf)
    ap.add_argument("--verify-below", type=float, default=p.verify_below,
                    help="only candidates with peak probability below this are verified")
    ap.add_argument("--min-chars", type=int, default=p.min_chars)
    ap.add_argument("--dilate", type=int, default=p.dilate)
    ap.add_argument("--save-prob", help="directory for probability maps")
    ap.add_argument("--debug-dir", help="directory for overlay images")
    ap.add_argument("--report", help="CSV with one row per candidate component")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    params = MaskParams(low_thresh=args.low_thresh, seed_thresh=args.seed_thresh,
                        min_area=args.min_area, min_conf=args.min_conf,
                        verify_below=args.verify_below, min_chars=args.min_chars,
                        dilate=args.dilate)
    det, rec, rec_name = build_models(args)
    print(f"detector: {det.name}, recognizer: {rec_name}, tile {args.tile}/{args.overlap}, {params}")

    root, out = Path(args.images), Path(args.out)
    files = list_images([args.images])
    report = None
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        report_fh = open(args.report, "w", newline="")
        report = csv.writer(report_fh)
        report.writerow(["file", "x", "y", "w", "h", "area", "peak", "text", "conf", "kept", "reason"])
    n_text = n_kept = n_rej = 0
    t0 = time.time()
    for i, f in enumerate(files, 1):
        rel = f.relative_to(root)
        dst = (out / rel).with_suffix(".png")
        if args.skip_existing and dst.exists():
            continue
        img = np.asarray(Image.open(f).convert("RGB"))
        prob = tiled_prob_map(img, det, args.tile, args.overlap)
        res = build_mask(img, prob, params, rec)
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(res.mask).save(dst)
        if args.save_prob:
            pp = (Path(args.save_prob) / rel).with_suffix(".png")
            pp.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((prob * 255 + 0.5).astype(np.uint8)).save(pp)
        if args.debug_dir:
            dp = (Path(args.debug_dir) / rel).with_suffix(".jpg")
            dp.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(overlay(img, res)).save(dp, quality=90)
        if report is not None:
            for c in res.components:
                if c.reason in ("small",):
                    continue
                report.writerow([str(rel), *c.box, c.area, f"{c.peak:.3f}", c.text,
                                 "" if c.conf != c.conf else f"{c.conf:.3f}", int(c.kept), c.reason])
        n_text += int(res.num_kept > 0)
        n_kept += res.num_kept
        n_rej += sum(c.reason == "not text" for c in res.components)
        print(f"[{i}/{len(files)}] {rel}: {res.num_kept} text regions, "
              f"{(res.mask > 0).mean() * 100:.2f}% of pixels", flush=True)
    if report is not None:
        report_fh.close()
    print(f"done in {time.time() - t0:.1f}s: {len(files)} masks, {n_text} images with text, "
          f"{n_kept} regions kept, {n_rej} rejected by the recognizer")


if __name__ == "__main__":
    main()
