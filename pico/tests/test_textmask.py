import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from picocodec.textmask import (  # noqa: E402
    MaskParams, PaddleDetector, PaddleRecognizer, build_mask, db_prob_to_image,
    tile_starts, tiled_prob_map,
)


def blob_prob(h=200, w=300):
    """Two 'text' blobs (strong, weak), one tiny blob."""
    p = np.zeros((h, w), np.float32)
    p[20:40, 20:120] = 0.95     # strong
    p[100:115, 50:150] = 0.6    # weak: above low_thresh, below seed_thresh
    p[150:152, 200:202] = 0.99  # tiny
    return p


def test_tile_starts_cover_everything():
    for n, tile, ov in [(100, 1024, 128), (1024, 1024, 128), (3000, 1024, 128), (1100, 512, 64)]:
        st = tile_starts(n, tile, ov)
        covered = np.zeros(n, bool)
        for s in st:
            covered[s:s + tile] = True
            assert 0 <= s and s + min(tile, n) <= n
        assert covered.all()


def test_tiled_prob_map_equals_full_map_for_pointwise_detector():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (700, 900, 3), dtype=np.uint8)

    def det(t):  # pointwise "detector": result independent of tiling
        return t[..., 0].astype(np.float32) / 255

    full = det(img)
    assert np.array_equal(tiled_prob_map(img, det, tile=256, overlap=32), full)


def test_build_mask_hysteresis_area_and_dilation():
    p = blob_prob()
    img = np.zeros(p.shape + (3,), np.uint8)
    res = build_mask(img, p, MaskParams(dilate=0))
    reasons = sorted(c.reason for c in res.components)
    assert reasons == ["", "small", "weak"]
    assert res.num_kept == 1
    assert (res.mask[20:40, 20:120] == 255).all() and res.mask.sum() == 255 * 20 * 100
    dil = build_mask(img, p, MaskParams(dilate=3)).mask
    assert dil[17, 70] == 255 and dil[16, 70] == 0  # grown by 3 px


def test_recognizer_filter_and_verify_below():
    p = blob_prob()
    p[60:80, 20:120] = 0.8   # second strong-ish candidate
    img = np.zeros(p.shape + (3,), np.uint8)
    calls = []

    def rec(crop):
        calls.append(crop.shape)
        return ("", 0.0) if len(calls) == 2 else ("TEXT", 0.9)

    res = build_mask(img, p, MaskParams(dilate=0), rec)
    assert len(calls) == 2 and res.num_kept == 1
    assert [c.reason for c in res.components if not c.kept and c.reason != "small"
            and c.reason != "weak"] == ["not text"]
    # verify_below=0.9: the 0.95 candidate is kept without asking the recognizer
    calls.clear()
    res = build_mask(img, p, MaskParams(dilate=0, verify_below=0.9), lambda c: ("", 0.0))
    assert res.num_kept == 1 and res.components[0].peak > 0.9


def test_db_prob_to_image():
    # source 100x150 resized by 0.64 -> 64x96 (multiples of 32), as DB does
    pred = np.zeros((1, 64, 96), np.float32)
    pred[0, 32:, :48] = 1.0
    out = db_prob_to_image(pred, [100, 150, 0.64, 0.64])
    assert out.shape == (100, 150)
    assert out[90, 10] > 0.99 and out[10, 140] < 0.01


class _FakeInner:
    """Mimics paddlex TextDetRunnerPredictor: calls post_op(preds, shapes)."""

    def __init__(self):
        self.post_op = lambda preds, shapes, **k: ([[np.array([[1, 1], [8, 1], [8, 4], [1, 4]])]], [[0.9]])

    def run(self, img, limit_side_len):
        h, w = img.shape[:2]
        pred = np.zeros((1, 1, h, w), np.float32)
        pred[0, 0, 1:5, 1:9] = 0.9
        polys, scores = self.post_op([pred], [np.array([h, w, 1.0, 1.0])])
        return [{"dt_polys": polys[0], "dt_scores": scores[0]}]


class _FakeTextDetection:
    def __init__(self, with_inner=True):
        if with_inner:
            self.paddlex_predictor = _FakeInner()
        self._inner = _FakeInner()

    def predict(self, input, limit_side_len, limit_type):
        assert limit_type == "max" and limit_side_len >= max(input.shape[:2])
        return (self.paddlex_predictor if hasattr(self, "paddlex_predictor")
                else self._inner).run(input, limit_side_len)


def test_paddle_detector_probability_hook_and_fallback():
    tile = np.zeros((16, 32, 3), np.uint8)
    det = PaddleDetector(predictor=_FakeTextDetection())
    assert det.prob_mode
    p = det(tile)
    assert p.shape == (16, 32) and p[2, 4] == pytest.approx(0.9) and p[10, 20] == 0
    fb = PaddleDetector(predictor=_FakeTextDetection(with_inner=False))
    assert not fb.prob_mode
    p = fb(tile)
    assert p[2, 4] == pytest.approx(0.9) and p[10, 20] == 0  # filled polygon


def test_paddle_recognizer_tries_rotations():
    class FakeRec:
        def predict(self, input):
            # only the view rotated to landscape reads well
            return [{"rec_text": "ok" if v.shape[1] > v.shape[0] else "", "rec_score":
                     0.9 if v.shape[1] > v.shape[0] else 0.1} for v in input]

    crop = np.zeros((40, 10, 3), np.uint8)  # vertical text
    assert PaddleRecognizer(predictor=FakeRec())(crop) == ("ok", 0.9)
    assert PaddleRecognizer(predictor=FakeRec(), rotations=())(crop) == ("", 0.1)


CRAFT_WEIGHTS = Path.home() / ".EasyOCR" / "model" / "craft_mlt_25k.pth"


@pytest.mark.skipif(not CRAFT_WEIGHTS.exists(), reason="CRAFT weights not downloaded")
def test_craft_finds_rendered_text():
    from PIL import Image, ImageDraw, ImageFont
    from picocodec.textmask import CraftDetector
    rng = np.random.default_rng(0)
    bg = (rng.normal(120, 10, (300, 600, 3))).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(bg)
    d = ImageDraw.Draw(im)
    d.text((40, 40), "HELLO WORLD", fill=(255, 255, 255), font=ImageFont.load_default(size=48))
    d.text((40, 200), "small text 123", fill=(0, 0, 0), font=ImageFont.load_default(size=16))
    img = np.asarray(im)
    res = build_mask(img, tiled_prob_map(img, CraftDetector(), tile=256, overlap=32), MaskParams())
    m = res.mask > 0
    assert m[50:85, 60:330].mean() > 0.5        # big text covered
    assert m[200:218, 45:150].mean() > 0.3      # small text covered
    assert m[110:180, :].mean() < 0.01          # background clean
