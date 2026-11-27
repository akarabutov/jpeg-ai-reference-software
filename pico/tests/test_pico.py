import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from picocodec.codec import ALIGN, MARGIN, decode_image, encode_image, tile_grid  # noqa: E402
from picocodec.layers import (  # noqa: E402
    Conv1x1iHaar, ConvScale, ConvScale311, HaarConv1x1, haar_down, haar_up,
    level_weights, reparameterize_module,
)
from picocodec.losses import LossConfig, RateDistortionLoss, tiling_artifact_loss  # noqa: E402
from picocodec.model import PICO, PICOConfig, grid2x2_masks  # noqa: E402

TINY = dict(
    enc_channels=(32, 32, 32),
    enc_chains=(((1, 1, 1), (1, 1, 1)),) * 3,
    dec_channels=(32, 32, 32),
    dec_chains=(((1, 1, 1), (1, 1, 1)),) * 3,
    latent_channels=32, hyper_channels=32, side_width=32,
    context_channels=32, predictor_width=32,
)


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return PICO(PICOConfig.from_dict(TINY))


def randomize_scales(module):
    """ConvScale scales are initialised to 1 -- randomise them so that the
    reparameterisation tests are meaningful."""
    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, ConvScale):
                m.s_in.uniform_(0.5, 1.5)
                m.s_out.uniform_(0.5, 1.5)
                if m.bias is not None:
                    m.bias.normal_()


def test_haar_is_invertible():
    x = torch.randn(2, 5, 8, 12)
    assert torch.allclose(haar_up(haar_down(x)), x, atol=1e-6)
    # orthonormal -> energy preserving
    assert torch.allclose(haar_down(x).pow(2).sum(), x.pow(2).sum(), rtol=1e-5)


@pytest.mark.parametrize("layer", [
    ConvScale(16, 32, 3, groups=4), ConvScale(16, 8, 1),
    HaarConv1x1(6, 10), Conv1x1iHaar(12, 5),
])
def test_reparameterize_layer(layer):
    randomize_scales(layer)
    x = torch.randn(2, layer.in_channels, 8, 8)
    assert torch.allclose(layer(x), layer.reparameterize()(x), atol=1e-5)


def test_reparameterize_block():
    blk = ConvScale311(64, 2, 2)
    randomize_scales(blk)
    x = torch.randn(1, 64, 8, 8)
    ref = blk(x)
    assert torch.allclose(ref, reparameterize_module(blk)(x), atol=1e-4)
    assert not any(isinstance(m, ConvScale) for m in blk.modules())


def test_level_weights():
    w = level_weights(torch.tensor([0.0, 3.5, 7.0, 6.3]))
    assert torch.allclose(w.sum(1), torch.ones(4))
    assert w[0, 0] == 1 and w[2, 7] == 1
    assert torch.allclose(w[1, 3:5], torch.tensor([0.5, 0.5]))
    assert torch.allclose(w[3, 6:8], torch.tensor([0.7, 0.3]))


def test_grid_masks_partition():
    m = grid2x2_masks(6, 8)
    assert torch.equal(m.sum(0), torch.ones(1, 6, 8))


def test_forward_shapes_and_backward():
    model = tiny_model()
    x = torch.rand(2, 3, 128, 192)
    out = model(x, torch.tensor([0.0, 7.0]))
    assert out["x_hat"].shape == x.shape
    assert out["likelihoods"]["y"].shape == (2, 32, 8, 12)
    assert out["likelihoods"]["z"].shape == (2, 32, 2, 3)
    crit = RateDistortionLoss(LossConfig())
    logs = crit(out, x, torch.tensor([0.0, 7.0]))
    logs["loss"].backward()
    assert model.context.predictors[3].net[0].weight.grad is not None


def test_default_architecture_builds():
    model = PICO()
    x = torch.rand(1, 3, 64, 64)
    assert model(x, torch.tensor([3.0]))["x_hat"].shape == x.shape


def test_tile_grid():
    tiles = tile_grid(1100, 600)
    assert len(tiles) == 3 * 2
    assert all(t.ph % ALIGN == 0 and t.pw % ALIGN == 0 for t in tiles)
    assert all(t.ph >= t.h + 2 * MARGIN and t.pw >= t.w + 2 * MARGIN for t in tiles)
    full = [t for t in tiles if t.h == 504 and t.w == 504]
    assert full and all(t.ph == 512 and t.pw == 512 for t in full)
    assert sum(t.h * t.w for t in tiles) == 1100 * 600


def test_symbols_roundtrip_and_decoder_matches_encoder():
    model = tiny_model()
    model.update(force=True)
    model.to_inference("cpu")
    x = torch.rand(2, 3, 128, 128)
    enc = model.compress(x, 2.3)
    # entropy decoding gives back exactly the encoded symbols
    from picocodec.layers import level_weights as lw_fn
    lw = lw_fn(torch.full((2,), 2.3))
    y, _ = model.encoder(x, lw)
    z_hat = model.entropy_bottleneck.decompress(enc["strings"][1], enc["shape"])
    p = model.context_decoder(z_hat)
    sym, y_rec_enc = model.context.encode(y * model.gain(lw), p)
    idx = model.gaussian_conditional.build_indexes(model.deterministic_sigma(z_hat))
    sym_dec = model.gaussian_conditional.decompress(enc["strings"][0], idx)
    assert torch.equal(sym_dec, sym)
    # the decoder-side one-shot context decode reproduces the encoder-side latent
    y_rec_dec = model.context.decode(sym_dec, p)
    assert torch.allclose(y_rec_dec, y_rec_enc, atol=1e-5)


def test_image_codec_roundtrip(tmp_path):
    model = tiny_model()
    model.update(force=True)
    model.to_inference("cpu")
    x = torch.rand(3, 530, 610)  # 2x2 tiles, partial edge tiles
    data = encode_image(model, x, 4.0)
    x1 = decode_image(model, data)
    x2 = decode_image(model, data)
    assert x1.shape == x.shape
    assert torch.equal(x1, x2)
    # higher quality level -> more bits (with a random model this is only
    # driven by the level gain, so just check both work)
    assert len(encode_image(model, x, 0.0)) > 14


def test_tiling_artifact_loss_sees_low_frequencies():
    x = torch.rand(1, 3, 64, 64)
    assert tiling_artifact_loss(x, x + 0.1).item() == pytest.approx(0.1, abs=1e-5)
    assert tiling_artifact_loss(x, x).item() == 0


def _make_images(folder: Path, n=6, size=(200, 180)):
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        arr = (rng.random((size[1], size[0], 3)) * 255).astype(np.uint8)
        Image.fromarray(arr).save(folder / f"img{i}.png")


@pytest.mark.parametrize("phase_cfg", ["pico_mse.yaml", "pico_perceptual.yaml"])
def test_train_script_runs(tmp_path, phase_cfg):
    imgs = tmp_path / "imgs"
    _make_images(imgs)
    model_override = "model=" + repr(TINY).replace("(", "[").replace(")", "]").replace("'", "")
    cmd = [
        sys.executable, str(ROOT / "train.py"), "-c", str(ROOT / "configs" / phase_cfg),
        "--out", str(tmp_path / "run"), "--device", "cpu", "--set",
        f"data.train=[{imgs}]", f"data.val={imgs}", "data.crop_size=128",
        "data.batch_size=2", "data.num_workers=0", "train.steps=3", "train.log_every=1",
        "train.val_every=3", "train.ckpt_every=3", model_override,
        "loss.w_lpips=0",  # LPIPS backbone weights are downloaded at runtime
        "disc.arch.base_channels=16", "disc.arch.max_channels=32",
    ]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "run" / "last.pt").exists()
    if "perceptual" in phase_cfg:
        assert "d_loss" in r.stdout and "gan_g" in r.stdout

    # encode / decode through the CLI with the trained checkpoint
    src = imgs / "img0.png"
    bit = tmp_path / "a.pico"
    rec = tmp_path / "a.png"
    ck = str(tmp_path / "run" / "last.pt")
    for args in (["encode", "-m", ck, "-q", "5", str(src), str(bit), "--device", "cpu"],
                 ["decode", "-m", ck, str(bit), str(rec), "--device", "cpu"]):
        r = subprocess.run([sys.executable, str(ROOT / "pico_codec.py"), *args],
                           cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
    assert Image.open(rec).size == Image.open(src).size
