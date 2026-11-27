"""Patch-wise discriminator (Sec. 4.2): PatchGAN of Isola et al. [40] with
increased capacity (more channels and conv layers).  The exact widths/depth
are not given in the paper; defaults below are our choice.  Spectral norm is
used for additional stability."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 128, num_layers: int = 4,
                 max_channels: int = 1024, extra_layers: int = 1, use_spectral_norm: bool = True):
        super().__init__()
        sn = spectral_norm if use_spectral_norm else (lambda m: m)
        layers = [sn(nn.Conv2d(in_channels, base_channels, 4, 2, 1)), nn.LeakyReLU(0.2, True)]
        ch = base_channels
        for i in range(1, num_layers):
            nxt = min(ch * 2, max_channels)
            stride = 2 if i < num_layers - 1 else 1
            layers += [sn(nn.Conv2d(ch, nxt, 4, stride, 1)), nn.LeakyReLU(0.2, True)]
            ch = nxt
        # extra stride-1 3x3 layers = "more convolution layers" than PatchGAN
        for _ in range(extra_layers):
            layers += [sn(nn.Conv2d(ch, ch, 3, 1, 1)), nn.LeakyReLU(0.2, True)]
        layers.append(sn(nn.Conv2d(ch, 1, 4, 1, 1)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x in [0, 1] -> patch logits [B, 1, h, w]."""
        return self.net(x * 2 - 1)


def _mask_to_logits(mask: Optional[torch.Tensor], logits: torch.Tensor) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    return F.adaptive_max_pool2d(mask, logits.shape[-2:])


def generator_loss(d_fake: torch.Tensor, text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Non-saturating generator loss per sample [B].  The GAN term is subdued
    in text-salient regions (weight 1 - m)."""
    loss = F.softplus(-d_fake)
    m = _mask_to_logits(text_mask, d_fake)
    if m is not None:
        loss = loss * (1 - m)
    return loss.flatten(1).mean(1)


def discriminator_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    return F.softplus(-d_real).mean() + F.softplus(d_fake).mean()
