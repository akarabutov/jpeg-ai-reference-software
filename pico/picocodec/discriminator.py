"""Adversarial part of PICO's perceptual training (Sec. 4.2).

Paper: "We use a patch-wise discriminator architecture similar to [40]
(PatchGAN), but boost the discriminator capacity by increasing the number of
channels and convolution layers.  However, a larger discriminator leads to
training instabilities, given that the lightweight decoder has limited
capacity."  Stabilisers named in the paper: two-stage training (MSE first) and
a warm-up that gradually increases the weight of the discriminator supervision.

Additional stabilisers implemented here (our choice, all configurable):
  * spectral normalisation in every discriminator conv;
  * optional R1 gradient penalty on real images (lazy, every N steps);
  * a delayed start of the GAN term for the codec, so that the discriminator
    gets a head start before it supervises the codec ("D warm-up");
  * non-saturating or hinge losses;
  * text-saliency masking of the generator loss (GAN subdued on text).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

GAN_LOSSES = ("ns", "hinge")


class PatchDiscriminator(nn.Module):
    """PatchGAN [Isola et al.] with more channels / layers.

    Default: receptive field 86 px, ~20.5M params; output: one logit per patch.
    """

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
    """Downsample a pixel mask to the patch grid (a patch counts as text if any
    of its pixels is text)."""
    if mask is None:
        return None
    return F.adaptive_max_pool2d(mask, logits.shape[-2:])


def generator_loss(d_fake: torch.Tensor, text_mask: Optional[torch.Tensor] = None,
                   loss_type: str = "ns") -> torch.Tensor:
    """Codec (generator) adversarial loss per sample [B].

    In text-salient patches the GAN term is switched off (weight 1 - m); the
    TextFidelityLoss supervises those regions instead.
    """
    if loss_type == "ns":
        loss = F.softplus(-d_fake)
    elif loss_type == "hinge":
        loss = -d_fake
    else:
        raise ValueError(f"unknown GAN loss {loss_type!r}, expected one of {GAN_LOSSES}")
    m = _mask_to_logits(text_mask, d_fake)
    if m is not None:
        loss = loss * (1 - m)
    return loss.flatten(1).mean(1)


def discriminator_loss(d_real: torch.Tensor, d_fake: torch.Tensor, loss_type: str = "ns") -> torch.Tensor:
    if loss_type == "ns":
        return F.softplus(-d_real).mean() + F.softplus(d_fake).mean()
    if loss_type == "hinge":
        return F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()
    raise ValueError(f"unknown GAN loss {loss_type!r}, expected one of {GAN_LOSSES}")


def r1_penalty(d_real: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
    """R1 = E ||grad_x D(x)||^2 on real images (x_real must require grad)."""
    (grad,) = torch.autograd.grad(d_real.sum(), x_real, create_graph=True)
    return grad.float().pow(2).flatten(1).sum(1).mean()
