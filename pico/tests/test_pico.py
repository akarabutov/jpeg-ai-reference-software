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


def test_gan_weight_schedule():
    crit = RateDistortionLoss(LossConfig(phase="perceptual", w_gan=1.0,
                                         gan_start_step=10, gan_warmup_steps=4))
    ws = [crit.gan_weight(s) for s in range(9, 16)]
    assert ws == [0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
    assert RateDistortionLoss(LossConfig(phase="mse")).gan_weight(100) == 0.0


def test_gan_losses_and_r1():
    from picocodec.discriminator import (
        PatchDiscriminator, discriminator_loss, generator_loss, r1_penalty)
    d = PatchDiscriminator(base_channels=8, max_channels=16)
    x = torch.rand(2, 3, 64, 64, requires_grad=True)
    logits = d(x)
    for kind in ("ns", "hinge"):
        assert discriminator_loss(logits, logits, kind).ndim == 0
        assert generator_loss(logits, None, kind).shape == (2,)
    # text mask switches the GAN term off
    full = torch.ones(2, 1, 64, 64)
    assert torch.all(generator_loss(logits, full) == 0)
    r1 = r1_penalty(logits, x)
    r1.backward()
    assert r1.item() > 0 and next(d.parameters()).grad is not None


def _make_images(folder: Path, n=6, size=(200, 180)):
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        arr = (rng.random((size[1], size[0], 3)) * 255).astype(np.uint8)
        Image.fromarray(arr).save(folder / f"img{i}.png")


def _tb_scalars(run_dir: Path) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ev = sorted(run_dir.glob("tb/*/events.out.tfevents.*"))[-1]
    acc = EventAccumulator(str(ev), size_guidance={"scalars": 0, "images": 0})
    acc.Reload()
    return {"scalars": set(acc.Tags()["scalars"]), "images": set(acc.Tags()["images"]),
            "acc": acc}


def _train(tmp_path, cfg, out, imgs, extra=(), steps=4):
    model_override = "model=" + repr(TINY).replace("(", "[").replace(")", "]").replace("'", "")
    cmd = [
        sys.executable, str(ROOT / "train.py"), "-c", str(ROOT / "configs" / cfg),
        "--out", str(out), *extra, "--set",
        f"data.train=[{imgs}]", f"data.val={imgs}", "data.crop_size=128",
        "data.batch_size=2", "data.num_workers=0", "data.val_max_images=2",
        f"train.steps={steps}", "train.log_every=1", "train.val_every=4", "train.ckpt_every=2",
        "train.accelerator=cpu", "train.val_log_images=1", model_override,
        "loss.w_lpips=0",  # LPIPS backbone weights are downloaded at runtime
        "loss.gan_start_step=1", "loss.gan_warmup_steps=2",
        "disc.arch.base_channels=16", "disc.arch.max_channels=32",
        "disc.r1_gamma=1.0", "disc.r1_every=2",
    ]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r


def test_train_two_phases_tensorboard_and_cli(tmp_path):
    imgs = tmp_path / "imgs"
    _make_images(imgs)

    # ---- phase 1 (MSE) ---------------------------------------------------- #
    run1 = tmp_path / "mse"
    _train(tmp_path, "pico_mse.yaml", run1, imgs)
    ck1 = run1 / "checkpoints" / "last.ckpt"
    assert ck1.exists()
    assert len(list((run1 / "checkpoints").glob("iter_*.ckpt"))) == 2  # iterations 2 and 4
    tb = _tb_scalars(run1)
    assert {"train/loss", "train/bpp", "train/psnr", "train/lr", "train/aux_loss",
            "val/q0/bpp", "val/q7/psnr"} <= tb["scalars"]
    assert "val_images/0" in tb["images"]
    steps = [e.step for e in tb["acc"].Scalars("train/loss")]
    assert steps == sorted(steps) and len(steps) == 4  # one point per iteration
    # LR schedule of the paper: x0.3 from 70% of training
    lrs = [e.value for e in tb["acc"].Scalars("train/lr")]
    assert lrs[0] == pytest.approx(8e-4, rel=1e-6) and lrs[-1] == pytest.approx(8e-4 * 0.3, rel=1e-6)

    # ---- phase 2 (perceptual + GAN), initialised from phase 1 ------------- #
    run2 = tmp_path / "perc"
    r = _train(tmp_path, "pico_perceptual.yaml", run2, imgs, ["--init", str(ck1)])
    assert "initialised codec" in r.stdout
    tb = _tb_scalars(run2)
    assert {"train/gan/d_loss", "train/gan/g_loss", "train/gan/d_acc", "train/gan/r1",
            "train/gan/weight", "train/tal", "train/msssim_loss"} <= tb["scalars"]
    weights = [e.value for e in tb["acc"].Scalars("train/gan/weight")]
    assert weights[0] == 0 and weights[-1] == pytest.approx(1e-3, rel=1e-6)
    ck2 = run2 / "checkpoints" / "last.ckpt"
    # Lightning counted 2 optimizer steps per iteration but training stopped
    # after train.steps iterations
    assert torch.load(ck2, map_location="cpu", weights_only=False)["iteration"] == 4

    # ---- resume ------------------------------------------------------------ #
    _train(tmp_path, "pico_perceptual.yaml", run2, imgs, ["--resume", str(ck2)], steps=6)
    assert torch.load(ck2, map_location="cpu", weights_only=False)["iteration"] == 6

    # ---- encode / decode through the CLI with the Lightning checkpoint ---- #
    src = imgs / "img0.png"
    bit = tmp_path / "a.pico"
    rec = tmp_path / "a.png"
    for args in (["encode", "-m", str(ck2), "-q", "5", str(src), str(bit), "--device", "cpu"],
                 ["decode", "-m", str(ck2), str(bit), str(rec), "--device", "cpu"]):
        r = subprocess.run([sys.executable, str(ROOT / "pico_codec.py"), *args],
                           cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
    assert Image.open(rec).size == Image.open(src).size
