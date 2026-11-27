"""Training objective of PICO (Sec. 3.3, 4.2, App. F).

    L = E_l [ alpha_l * D(x, x^_l) + alpha_l * lambda_l * R(y^_hyper_l, y^_l) ]

    D = MSE + w1 LPIPS + w2 (1 - MS-SSIM) + w3 TilingArtifactLoss
        + w4 TextFidelityLoss(m) + w5 GAN(m)

Phase 1 ("mse") uses D = MSE only, phase 2 ("perceptual") uses all terms.
All distortion terms are computed per sample so that the level-dependent
weights alpha_l, lambda_l can be applied per sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import NUM_COARSE_LEVELS, interp_table, level_weights

# CompressAI's MSE lambdas (L = lambda * 255^2 * MSE + bpp) for its 8 quality
# levels, re-expressed as a multiplier on the rate with MSE on [0, 1]:
#   lambda_R = 1 / (lambda * 255^2).   Level 0 = lowest quality.
_COMPRESSAI_LAMBDAS = (0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483, 0.0932, 0.1800)
DEFAULT_RATE_LAMBDAS = tuple(1.0 / (lam * 255.0 ** 2) for lam in _COMPRESSAI_LAMBDAS)


def bits_per_pixel(likelihoods: Dict[str, torch.Tensor], num_pixels: int) -> torch.Tensor:
    """Per-sample rate in bpp, [B]."""
    bits = sum((-torch.log2(l)).flatten(1).sum(1) for l in likelihoods.values())
    return bits / num_pixels


def tiling_artifact_loss(x: torch.Tensor, x_hat: torch.Tensor,
                         factors: Sequence[int] = (2, 4, 8, 16, 32)) -> torch.Tensor:
    """TilingArtifactLoss: multi-resolution L1 that supervises low spatial
    frequencies (colour / DC consistency) which perceptual and GAN losses tend
    to ignore, [B]."""
    losses = []
    for f in factors:
        if x.shape[-1] < f or x.shape[-2] < f:
            continue
        losses.append((F.avg_pool2d(x, f) - F.avg_pool2d(x_hat, f)).abs().flatten(1).mean(1))
    return torch.stack(losses, 0).mean(0)


def text_fidelity_loss(x: torch.Tensor, x_hat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """TextFidelityLoss: heavy L1 inside text-saliency regions, [B]."""
    return ((x - x_hat).abs() * mask).flatten(1).mean(1)


class PerceptualMetrics(nn.Module):
    """Lazily constructed LPIPS / MS-SSIM (only needed in phase 2)."""

    def __init__(self, lpips_net: str = "vgg"):
        super().__init__()
        self.lpips_net = lpips_net
        self._lpips = None

    def lpips(self, x, x_hat):
        if self._lpips is None:
            import lpips  # noqa: WPS433
            self._lpips = lpips.LPIPS(net=self.lpips_net, verbose=False).to(x.device).eval()
            for p in self._lpips.parameters():
                p.requires_grad_(False)
        return self._lpips(x * 2 - 1, x_hat * 2 - 1, normalize=False).flatten(1).mean(1)

    @staticmethod
    def ms_ssim_loss(x, x_hat):
        from pytorch_msssim import ms_ssim
        # 5 scales with an 11x11 window need min side > 160 px; shrink the window
        # for smaller training crops.
        side = min(x.shape[-2:])
        win = 11
        while win > 3 and side <= (win - 1) * 16:
            win -= 2
        val = ms_ssim(x.float(), x_hat.float(), data_range=1.0, size_average=False, win_size=win)
        return 1.0 - val


@dataclass
class LossConfig:
    phase: str = "mse"                       # "mse" | "perceptual"
    rate_lambdas: Sequence[float] = DEFAULT_RATE_LAMBDAS   # lambda_l, one per coarse level
    alphas: Sequence[float] = (1.0,) * NUM_COARSE_LEVELS   # alpha_l
    w_lpips: float = 1.0 / 150             # w1   (weights: not given in the paper)
    w_msssim: float = 1.0 / 150            # w2
    w_tiling: float = 0.02                 # w3
    w_text: float = 0.1                    # w4
    w_gan: float = 1e-3                    # w5 (final value after warm-up)
    gan_warmup_steps: int = 20_000
    tal_factors: Sequence[int] = (2, 4, 8, 16, 32)
    lpips_net: str = "vgg"

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "LossConfig":
        return cls(**dict(d or {}))


class RateDistortionLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg
        assert len(cfg.rate_lambdas) == NUM_COARSE_LEVELS
        assert len(cfg.alphas) == NUM_COARSE_LEVELS
        self.register_buffer("rate_lambdas", torch.tensor(cfg.rate_lambdas, dtype=torch.float32))
        self.register_buffer("alphas", torch.tensor(cfg.alphas, dtype=torch.float32))
        self.metrics = PerceptualMetrics(cfg.lpips_net)

    def gan_weight(self, step: int) -> float:
        if self.cfg.phase != "perceptual":
            return 0.0
        warm = max(1, self.cfg.gan_warmup_steps)
        return self.cfg.w_gan * min(1.0, step / warm)

    def forward(
        self,
        out: Dict,
        x: torch.Tensor,
        level: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        gan_g_loss: Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """`gan_g_loss`: per-sample generator loss [B] (already mask-weighted)."""
        x_hat = out["x_hat"]
        num_pixels = x.shape[-2] * x.shape[-1]
        lw = level_weights(level)
        lam = interp_table(self.rate_lambdas, lw)    # geometric interpolation
        alpha = interp_table(self.alphas, lw)

        bpp = bits_per_pixel(out["likelihoods"], num_pixels)
        mse = (x - x_hat).pow(2).flatten(1).mean(1)
        logs = {"bpp": bpp.mean(), "mse": mse.mean(),
                "psnr": (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).mean()}
        dist = mse

        if self.cfg.phase == "perceptual":
            c = self.cfg
            if c.w_lpips > 0:
                lp = self.metrics.lpips(x, x_hat)
                dist = dist + c.w_lpips * lp
                logs["lpips"] = lp.mean()
            if c.w_msssim > 0:
                ms = self.metrics.ms_ssim_loss(x, x_hat)
                dist = dist + c.w_msssim * ms
                logs["msssim_loss"] = ms.mean()
            if c.w_tiling > 0:
                tal = tiling_artifact_loss(x, x_hat, c.tal_factors)
                dist = dist + c.w_tiling * tal
                logs["tal"] = tal.mean()
            if text_mask is not None and c.w_text > 0:
                txt = text_fidelity_loss(x, x_hat, text_mask)
                dist = dist + c.w_text * txt
                logs["text_l1"] = txt.mean()
            if gan_g_loss is not None:
                wg = self.gan_weight(step)
                dist = dist + wg * gan_g_loss
                logs["gan_g"] = gan_g_loss.mean()
                logs["w_gan"] = torch.tensor(wg)

        loss = (alpha * (dist + lam * bpp)).mean()
        logs["loss"] = loss
        logs["distortion"] = dist.mean()
        return logs
