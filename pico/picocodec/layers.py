"""Building blocks of PICO ("What Matters in Practical Learned Image Compression",
arXiv:2605.05148).

* ConvScale            -- convolution with learned input/output scales (Sec. 4.1).
* LearnedScale         -- learned element-wise (per-channel) scale at the end of
                          every spatial-resolution block (Sec. 4.1, Fig. 4).
* HaarConv1x1          -- Haar-wavelet downsampling followed by a 1x1 conv.
* Conv1x1iHaar         -- 1x1 conv followed by inverse-Haar upsampling.
                          Both collapse to a single conv at inference (App. H).
* ConvScale311 / CSChain -- inverted-residual backbone block (Fig. 4, right).
* level helpers        -- quality-level one-hot interpolation (App. F).

Every layer that has a "training" parameterisation (extra scales, explicit
Haar) exposes ``reparameterize()`` that returns a plain, computationally
identical ``nn.Module`` for deployment.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Appendix B: "all 3x3 convolutions and ConvScales in the paper are configured to
# have the number of channels per group be 32, unless stated otherwise".  We
# apply grouping to 3x3 convolutions; 1x1 convolutions stay dense (grouping them
# too would remove almost all channel mixing).  Configurable via `group_size`.
DEFAULT_GROUP_SIZE = 32


def num_groups(c_in: int, c_out: int, group_size: Optional[int]) -> int:
    """Largest group count <= c_in // group_size that divides both channel counts."""
    if not group_size:
        return 1
    g = max(1, c_in // group_size)
    while c_in % g or c_out % g:
        g -= 1
    return g


# --------------------------------------------------------------------------- #
# ConvScale
# --------------------------------------------------------------------------- #
class ConvScale(nn.Module):
    """Convolution reparameterised with learned input and output scales.

    W' = s_in * s_out * W,  b' = squeeze(s_out) * b
    with s_in of shape [1, C//G, 1, 1] and s_out of shape [K, 1, 1, 1].
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        groups: int = 1,
        bias: bool = True,
        out_scale_init: float = 1.0,
    ):
        super().__init__()
        assert in_channels % groups == 0 and out_channels % groups == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)
        self.s_in = nn.Parameter(torch.ones(1, in_channels // groups, 1, 1))
        self.s_out = nn.Parameter(torch.full((out_channels, 1, 1, 1), float(out_scale_init)))

    def effective_weight(self) -> torch.Tensor:
        return self.weight * self.s_in * self.s_out

    def effective_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        return self.bias * self.s_out.view(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x, self.effective_weight(), self.effective_bias(),
            padding=self.padding, groups=self.groups,
        )

    @torch.no_grad()
    def reparameterize(self) -> nn.Conv2d:
        conv = nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size,
            padding=self.padding, groups=self.groups, bias=self.bias is not None,
        )
        conv.weight.copy_(self.effective_weight())
        if self.bias is not None:
            conv.bias.copy_(self.effective_bias())
        return conv.to(self.weight.device, self.weight.dtype)


class LearnedScale(nn.Module):
    """Learned per-channel multiplicative scale (initialised to 1)."""

    def __init__(self, channels: int, init: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), float(init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


# --------------------------------------------------------------------------- #
# Haar resampling
# --------------------------------------------------------------------------- #
def haar_matrix(device=None, dtype=None) -> torch.Tensor:
    """Orthonormal 2D Haar transform of a 2x2 block [a, b, c, d] (row-major).

    Rows: LL, LH, HL, HH.  The matrix is symmetric and orthogonal, i.e. it is
    its own inverse.
    """
    m = torch.tensor(
        [[1, 1, 1, 1],
         [1, 1, -1, -1],
         [1, -1, 1, -1],
         [1, -1, -1, 1]],
        dtype=dtype or torch.float32, device=device,
    )
    return m * 0.5


def haar_down(x: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [B, 4C, H/2, W/2]; channel order c*4 + subband."""
    b, c, h, w = x.shape
    u = F.pixel_unshuffle(x, 2).view(b, c, 4, h // 2, w // 2)
    m = haar_matrix(x.device, x.dtype)
    return torch.einsum("mk,bckhw->bcmhw", m, u).reshape(b, 4 * c, h // 2, w // 2)


def haar_up(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`haar_down`."""
    b, c4, h, w = x.shape
    c = c4 // 4
    m = haar_matrix(x.device, x.dtype)  # self-inverse
    u = torch.einsum("km,bcmhw->bckhw", m, x.view(b, c, 4, h, w)).reshape(b, c4, h, w)
    return F.pixel_shuffle(u, 2)


class HaarConv1x1(nn.Module):
    """2x downsampling: Haar transform followed by a 1x1 ConvScale (4*C_in -> C_out)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.conv = ConvScale(4 * in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(haar_down(x))

    @torch.no_grad()
    def reparameterize(self) -> nn.Module:
        """Collapse Haar into the 1x1 conv -> a single 2x2 / stride-2 conv."""
        w = self.conv.effective_weight()[:, :, 0, 0]  # [O, 4C] (index c*4+m)
        o = w.shape[0]
        w = w.view(o, self.in_channels, 4)  # [O, C, m]
        m = haar_matrix(w.device, w.dtype)
        wf = torch.einsum("ocm,mk->ock", w, m)  # [O, C, k], k = i*2 + j
        conv = nn.Conv2d(self.in_channels, o, 2, stride=2, bias=True)
        conv.weight.copy_(wf.view(o, self.in_channels, 2, 2))
        b = self.conv.effective_bias()
        conv.bias.copy_(b if b is not None else torch.zeros(o))
        return conv.to(w.device, w.dtype)


class _ConvPixelShuffle(nn.Sequential):
    pass


class Conv1x1iHaar(nn.Module):
    """2x upsampling: 1x1 ConvScale (C_in -> 4*C_out) followed by inverse Haar."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.conv = ConvScale(in_channels, 4 * out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return haar_up(self.conv(x))

    @torch.no_grad()
    def reparameterize(self) -> nn.Module:
        """Collapse inverse Haar into the conv -> 1x1 conv + PixelShuffle(2)."""
        w = self.conv.effective_weight()[:, :, 0, 0]  # [4*O, C]
        b = self.conv.effective_bias()
        o = self.out_channels
        m = haar_matrix(w.device, w.dtype)
        wf = torch.einsum("km,omc->okc", m, w.view(o, 4, -1)).reshape(4 * o, -1)
        conv = nn.Conv2d(self.in_channels, 4 * o, 1, bias=True)
        conv.weight.copy_(wf[:, :, None, None])
        if b is not None:
            conv.bias.copy_(torch.einsum("km,om->ok", m, b.view(o, 4)).reshape(-1))
        else:
            conv.bias.zero_()
        return _ConvPixelShuffle(conv, nn.PixelShuffle(2)).to(w.device, w.dtype)


# --------------------------------------------------------------------------- #
# Backbone blocks
# --------------------------------------------------------------------------- #
class ConvScale311(nn.Module):
    """Inverted residual block (Fig. 4, right):

    x + ConvScale1x1(C*F -> C)( GELU( ConvScale1x1(C*E -> C*F)( GELU( ConvScale3x3(C -> C*E)(x) ))))
    """

    def __init__(self, channels: int, expand3: int = 1, expand1: int = 1,
                 group_size: Optional[int] = DEFAULT_GROUP_SIZE,
                 residual_scale_init: float = 0.1):
        super().__init__()
        c, ce, cf = channels, channels * expand3, channels * expand1
        self.conv3 = ConvScale(c, ce, 3, groups=num_groups(c, ce, group_size))
        self.conv1a = ConvScale(ce, cf, 1)
        # Small initial output scale keeps the residual branch close to identity
        # at init (stabilises the start of training; not specified in the paper).
        self.conv1b = ConvScale(cf, c, 1, out_scale_init=residual_scale_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.conv3(x))
        h = F.gelu(self.conv1a(h))
        return x + self.conv1b(h)


class CSChain(nn.Sequential):
    """CS-Chain(C, R, E, F): R repetitions of ConvScale311(C, E, F)."""

    def __init__(self, channels: int, repeats: int, expand3: int, expand1: int,
                 group_size: Optional[int] = DEFAULT_GROUP_SIZE):
        super().__init__(*[
            ConvScale311(channels, expand3, expand1, group_size) for _ in range(repeats)
        ])


def reparameterize_module(module: nn.Module) -> nn.Module:
    """Recursively replace ConvScale / HaarConv1x1 / Conv1x1iHaar by plain ops.

    Returns the (in-place modified) module.  Outputs are numerically identical
    up to floating-point rounding.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, (ConvScale, HaarConv1x1, Conv1x1iHaar)):
            setattr(module, name, child.reparameterize())
        else:
            reparameterize_module(child)
    return module


# --------------------------------------------------------------------------- #
# Quality levels (Appendix F)
# --------------------------------------------------------------------------- #
NUM_COARSE_LEVELS = 8          # l_c in {0, ..., 7}
FINE_PER_COARSE = 10           # level density increased 10x
NUM_FINE_LEVELS = (NUM_COARSE_LEVELS - 1) * FINE_PER_COARSE + 1  # 71


def fine_index_to_level(idx):
    """Fine level index in [0, 70] -> continuous level in [0, 7]."""
    return idx / FINE_PER_COARSE


def level_to_fine_index(level: float) -> int:
    idx = int(round(float(level) * FINE_PER_COARSE))
    return max(0, min(NUM_FINE_LEVELS - 1, idx))


def level_weights(level: torch.Tensor) -> torch.Tensor:
    """Interpolated one-hot vectors, [B] (levels in [0, 7]) -> [B, 8]."""
    level = level.float().clamp(0, NUM_COARSE_LEVELS - 1)
    lo = level.floor().long().clamp(max=NUM_COARSE_LEVELS - 2)
    frac = (level - lo.float()).unsqueeze(1)
    w = torch.zeros(level.shape[0], NUM_COARSE_LEVELS, device=level.device)
    w.scatter_(1, lo.unsqueeze(1), 1 - frac)
    w.scatter_add_(1, (lo + 1).unsqueeze(1), frac)
    return w


def concat_level(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """ConcatLevel: append the spatially broadcast level vector [B, 8] to x."""
    b, _, h, ww = x.shape
    return torch.cat([x, w.to(x.dtype)[:, :, None, None].expand(b, -1, h, ww)], dim=1)


def interp_table(table: torch.Tensor, w: torch.Tensor, log_space: bool = True) -> torch.Tensor:
    """Interpolate a per-coarse-level table [8, ...] with level weights [B, 8]."""
    flat = table.reshape(table.shape[0], -1)
    if log_space:
        out = torch.exp(w @ torch.log(flat))
    else:
        out = w @ flat
    return out.reshape(w.shape[0], *table.shape[1:])
