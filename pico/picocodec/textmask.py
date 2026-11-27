"""Text saliency masks for TextFidelityLoss (Sec. 4.2).

Pipeline (per image, at the ORIGINAL resolution):

  1. probability map  -- a text detector is run on overlapping tiles (no
     downscaling, so small text survives) and the per-pixel text probability
     is merged with a max over the overlaps:
       * CRAFT  (the detector used in the paper; weights from EasyOCR):
                p = max(region score, affinity/link score)
       * PaddleOCR DB detector (PP-OCR det): DB probability map
  2. components       -- connected components of p > low_thresh; a component is
     kept as a candidate if its peak probability >= seed_thresh and its area
     >= min_area (the hysteresis CRAFT itself uses for its boxes).
  3. optional recognition filter -- candidates (with peak < verify_below) are
     cropped and read by a text recognizer (EasyOCR or PaddleOCR, also rotated
     by 90/180/270 degrees); candidates whose confidence is below min_conf or
     that contain fewer than min_chars alphanumeric characters are dropped.
     Meant against false positives on textures, but it also drops real text the
     recognizer cannot read (other scripts, handwriting, stylised logos,
     multi-line components), so it is OFF by default -- check with
     --debug-dir / --report before enabling it on a dataset.
  4. the kept components are dilated by `dilate` pixels -> binary mask.

Library-specific code is isolated in the Detector / Recognizer classes, so the
mask logic can be tested without model weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Detectors: RGB uint8 tile [h, w, 3] -> text probability float32 [h, w]
# --------------------------------------------------------------------------- #
class Detector:
    name = "base"

    def __call__(self, tile: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class CraftDetector(Detector):
    """CRAFT (Baek et al., CVPR 2019) through EasyOCR's packaged weights."""

    name = "craft"

    def __init__(self, reader=None, gpu: bool = False, langs: Sequence[str] = ("en",)):
        import torch
        if reader is None:
            import easyocr
            reader = easyocr.Reader(list(langs), gpu=gpu, verbose=False)
        self.reader = reader
        self.net = reader.detector
        self.device = reader.device
        self.torch = torch

    def __call__(self, tile: np.ndarray) -> np.ndarray:
        from easyocr.imgproc import normalizeMeanVariance
        h, w = tile.shape[:2]
        ph, pw = int(math.ceil(h / 32) * 32), int(math.ceil(w / 32) * 32)
        img = np.zeros((ph, pw, 3), np.uint8)
        img[:h, :w] = tile
        x = np.transpose(normalizeMeanVariance(img), (2, 0, 1))[None]
        with self.torch.no_grad():
            y, _ = self.net(self.torch.from_numpy(x).float().to(self.device))
        y = y[0].float().cpu().numpy()                 # [ph/2, pw/2, 2]
        p = np.maximum(y[..., 0], y[..., 1])           # region, affinity
        p = cv2.resize(p, (pw, ph), interpolation=cv2.INTER_LINEAR)
        return np.clip(p[:h, :w], 0, 1).astype(np.float32)


def db_prob_to_image(pred: np.ndarray, img_shape: Sequence[float]) -> np.ndarray:
    """PaddleOCR DB output [1, Hr, Wr] (on the resized input) + its shape record
    [src_h, src_w, ratio_h, ratio_w] -> probability map at the source size."""
    src_h, src_w, ratio_h, ratio_w = [float(v) for v in img_shape]
    p = np.asarray(pred, np.float32).reshape(pred.shape[-2], pred.shape[-1])
    rh = min(p.shape[0], int(round(src_h * ratio_h)))
    rw = min(p.shape[1], int(round(src_w * ratio_w)))
    p = cv2.resize(p[:rh, :rw], (int(src_w), int(src_h)), interpolation=cv2.INTER_LINEAR)
    return np.clip(p, 0, 1).astype(np.float32)


class PaddleDetector(Detector):
    """PaddleOCR (PP-OCR, DB) text detector.

    PaddleOCR only returns polygons; the DB probability map is captured by
    wrapping the predictor's DB post-processing step.  If that internal hook is
    unavailable (other PaddleOCR version), the detected polygons are filled
    instead and a warning is printed once.
    """

    name = "paddle"

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None,
                 predictor=None):
        if predictor is None:
            from paddleocr import TextDetection
            kwargs = {"model_name": model_name} if model_name else {}
            if device:
                kwargs["device"] = device
            predictor = TextDetection(**kwargs)
        self.predictor = predictor
        self._captured: List = []
        self.prob_mode = self._install_hook()
        if not self.prob_mode:
            print("warning: PaddleOCR probability map not accessible; filling polygons instead")

    def _install_hook(self) -> bool:
        inner = getattr(self.predictor, "paddlex_predictor", None)
        post = getattr(inner, "post_op", None)
        if post is None:
            return False

        def wrapped(preds, img_shapes, *a, **k):
            self._captured.append((np.asarray(preds[0]), [np.asarray(s) for s in img_shapes]))
            return post(preds, img_shapes, *a, **k)

        inner.post_op = wrapped
        return True

    def __call__(self, tile: np.ndarray) -> np.ndarray:
        h, w = tile.shape[:2]
        bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        self._captured.clear()
        # limit_type=max with limit = tile side -> no downscaling of the tile
        res = list(self.predictor.predict(input=bgr, limit_side_len=max(h, w), limit_type="max"))
        if self.prob_mode and self._captured:
            preds, shapes = self._captured[-1]
            return db_prob_to_image(preds[0], shapes[0])
        prob = np.zeros((h, w), np.float32)
        for r in res:
            for poly, score in zip(r["dt_polys"], r["dt_scores"]):
                cv2.fillPoly(prob, [np.asarray(poly, np.int32).reshape(-1, 2)], float(score))
        return prob


# --------------------------------------------------------------------------- #
# Recognizers: RGB uint8 crop -> (text, confidence)
# --------------------------------------------------------------------------- #
class Recognizer:
    name = "base"

    def __call__(self, crop: np.ndarray) -> Tuple[str, float]:  # pragma: no cover
        raise NotImplementedError


ROTATIONS = (90, 180, 270)


class EasyOCRRecognizer(Recognizer):
    """EasyOCR recognizer; also tries the crop rotated by `rotations` degrees
    (vertical / upside-down text) and returns the most confident reading."""

    name = "easyocr"

    def __init__(self, reader=None, gpu: bool = False, langs: Sequence[str] = ("en",),
                 rotations: Sequence[int] = ROTATIONS):
        if reader is None:
            import easyocr
            reader = easyocr.Reader(list(langs), gpu=gpu, verbose=False)
        self.reader = reader
        self.rotations = list(rotations) or None

    def __call__(self, crop: np.ndarray) -> Tuple[str, float]:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        out = self.reader.recognize(gray, horizontal_list=[[0, w, 0, h]], free_list=[], detail=1,
                                    rotation_info=self.rotations)
        if not out:
            return "", 0.0
        best = max(out, key=lambda o: float(o[2]))
        return str(best[1]), float(best[2])


class PaddleRecognizer(Recognizer):
    name = "paddle"

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None,
                 predictor=None, rotations: Sequence[int] = ROTATIONS):
        if predictor is None:
            from paddleocr import TextRecognition
            kwargs = {"model_name": model_name} if model_name else {}
            if device:
                kwargs["device"] = device
            predictor = TextRecognition(**kwargs)
        self.predictor = predictor
        self.rotations = list(rotations)

    def __call__(self, crop: np.ndarray) -> Tuple[str, float]:
        bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        views = [bgr] + [np.ascontiguousarray(np.rot90(bgr, k=-(r // 90))) for r in self.rotations]
        best = ("", 0.0)
        for res in self.predictor.predict(input=views):
            cand = (str(res["rec_text"]), float(res["rec_score"]))
            if cand[1] > best[1]:
                best = cand
        return best


# --------------------------------------------------------------------------- #
# Mask construction
# --------------------------------------------------------------------------- #
def tile_starts(n: int, tile: int, overlap: int) -> List[int]:
    if n <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, n - tile + 1, step))
    if starts[-1] != n - tile:
        starts.append(n - tile)
    return starts


def tiled_prob_map(img: np.ndarray, detector: Callable[[np.ndarray], np.ndarray],
                   tile: int = 1024, overlap: int = 128) -> np.ndarray:
    """Run the detector on overlapping tiles at full resolution; max-merge."""
    h, w = img.shape[:2]
    prob = np.zeros((h, w), np.float32)
    for y0 in tile_starts(h, tile, overlap):
        for x0 in tile_starts(w, tile, overlap):
            t = img[y0:y0 + tile, x0:x0 + tile]
            p = detector(t)
            assert p.shape == t.shape[:2], (p.shape, t.shape)
            np.maximum(prob[y0:y0 + t.shape[0], x0:x0 + t.shape[1]], p,
                       out=prob[y0:y0 + t.shape[0], x0:x0 + t.shape[1]])
    return prob


@dataclass
class Component:
    label: int
    box: Tuple[int, int, int, int]  # x, y, w, h
    area: int
    peak: float
    text: str = ""
    conf: float = float("nan")
    kept: bool = False
    reason: str = ""


@dataclass
class MaskParams:
    low_thresh: float = 0.4     # component extent
    seed_thresh: float = 0.7    # min peak probability of a component
    min_area: int = 20          # px, at full resolution
    min_conf: float = 0.3       # recognizer confidence
    verify_below: float = 1.0   # only candidates with peak < this are verified
    min_chars: int = 1          # alphanumeric characters in the recognized text
    crop_pad: int = 4           # px around a component for recognition
    dilate: int = 4             # px added around kept components


@dataclass
class MaskResult:
    mask: np.ndarray                      # uint8 {0, 255}
    prob: np.ndarray                      # float32 [0, 1]
    components: List[Component] = field(default_factory=list)

    @property
    def num_kept(self) -> int:
        return sum(c.kept for c in self.components)

    @property
    def num_rejected(self) -> int:
        return sum(not c.kept for c in self.components)


def build_mask(img: np.ndarray, prob: np.ndarray, params: MaskParams,
               recognizer: Optional[Callable[[np.ndarray], Tuple[str, float]]] = None
               ) -> MaskResult:
    h, w = prob.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (prob > params.low_thresh).astype(np.uint8), connectivity=8)
    peaks = np.zeros(n, np.float32)
    np.maximum.at(peaks, labels.ravel(), prob.ravel())
    comps: List[Component] = []
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, cw, ch, area = (int(v) for v in stats[i])
        c = Component(i, (x, y, cw, ch), area, float(peaks[i]))
        comps.append(c)
        if area < params.min_area:
            c.reason = "small"
            continue
        if c.peak < params.seed_thresh:
            c.reason = "weak"
            continue
        if recognizer is not None and c.peak < params.verify_below:
            p = params.crop_pad
            crop = img[max(0, y - p):min(h, y + ch + p), max(0, x - p):min(w, x + cw + p)]
            c.text, c.conf = recognizer(crop)
            n_alnum = sum(ch_.isalnum() for ch_ in c.text)
            if c.conf < params.min_conf or n_alnum < params.min_chars:
                c.reason = "not text"
                continue
        c.kept = True
        keep[i] = True
    mask = keep[labels].astype(np.uint8) * 255
    if params.dilate > 0 and mask.any():
        k = 2 * params.dilate + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return MaskResult(mask, prob, comps)


def overlay(img: np.ndarray, res: MaskResult) -> np.ndarray:
    """Debug view: kept mask in red, rejected candidates outlined in blue."""
    out = img.copy()
    m = res.mask > 0
    out[m] = (0.5 * out[m] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    for c in res.components:
        if not c.kept and c.reason == "not text":
            x, y, cw, ch = c.box
            cv2.rectangle(out, (x, y), (x + cw, y + ch), (0, 128, 255), 2)
    return out
