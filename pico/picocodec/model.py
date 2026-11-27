"""PICO model (Fig. 3, 4, 11 of arXiv:2605.05148).

    x, l --> OuterEncoder --> y, y_hyper
    y_hyper --> factorized entropy model --> y^_hyper
    y^_hyper --> ScaleDecoder   --> sigma   (bit-exact path, runs on CPU)
    y^_hyper --> ContextDecoder --> p
    (y * gain(l), p) --> one-shot context encode --> integer symbols y^ (coded with sigma)
    (y^, p) --> one-shot context decode --> y~ / gain(l) --> OuterDecoder(l) --> x^

Shapes: y is at 1/16 of the input resolution, y_hyper at 1/64.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models.utils import update_registered_buffers

from .layers import (
    NUM_COARSE_LEVELS,
    Conv1x1iHaar,
    ConvScale,
    CSChain,
    HaarConv1x1,
    LearnedScale,
    concat_level,
    interp_table,
    level_weights,
    reparameterize_module,
)

# (R, E, F) for the two CS-Chains of each spatial scale.
ChainSpec = Tuple[Tuple[int, int, int], Tuple[int, int, int]]


@dataclass
class PICOConfig:
    # Table 3 (a): outer encoder, final values.
    enc_channels: Tuple[int, int, int] = (64, 96, 96)
    enc_chains: Tuple[ChainSpec, ChainSpec, ChainSpec] = (
        ((1, 1, 2), (1, 4, 2)),
        ((2, 1, 1), (3, 4, 1)),
        ((2, 1, 1), (4, 1, 1)),
    )
    # Table 3 (b): outer decoder, final values.
    dec_channels: Tuple[int, int, int] = (160, 64, 32)
    dec_chains: Tuple[ChainSpec, ChainSpec, ChainSpec] = (
        ((3, 1, 1), (2, 1, 2)),
        ((2, 1, 1), (1, 4, 2)),
        ((2, 2, 2), (1, 3, 2)),
    )
    # Not stated in the paper (C in Fig. 4/11).  192 is a common choice.
    latent_channels: int = 192
    # Fig. 11: the scale decoder input is ConvScale1x1(192, 128).
    hyper_channels: int = 192
    side_width: int = 128          # width of scale / context decoders (Fig. 11)
    context_channels: int = 128    # channels of the prior p (not stated)
    predictor_width: int = 128     # width of Predictor_n (not stated)
    group_size: int = 32           # channels per group of 3x3 convs (App. B)
    scale_bound: float = 0.11
    q_min: float = 0.05            # lower bound of the learned quantization width

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "PICOConfig":
        d = dict(d or {})

        def tup(v):
            return tuple(tup(i) for i in v) if isinstance(v, (list, tuple)) else v

        return cls(**{k: tup(v) for k, v in d.items()})

    def to_dict(self) -> dict:
        return asdict(self)


def _stage(channels: int, chains: ChainSpec, group_size: int) -> nn.Sequential:
    (r1, e1, f1), (r2, e2, f2) = chains
    return nn.Sequential(
        CSChain(channels, r1, e1, f1, group_size),
        CSChain(channels, r2, e2, f2, group_size),
        LearnedScale(channels),
    )


# --------------------------------------------------------------------------- #
# Outer encoder (Fig. 11, left) with the absorbed hyper-encoder
# --------------------------------------------------------------------------- #
class OuterEncoder(nn.Module):
    def __init__(self, cfg: PICOConfig):
        super().__init__()
        c1, c2, c3 = cfg.enc_channels
        gs = cfg.group_size
        # "Haar(3, C1)" + "ConcatLevel": the level one-hot is concatenated to the
        # input and the Haar transform + 1x1 conv is a single HaarConv1x1.  Since
        # the level map is spatially constant, this is equivalent to concatenating
        # it after the Haar transform (Haar of a constant is LL-only).
        self.inp = HaarConv1x1(3 + NUM_COARSE_LEVELS, c1)
        self.stage1 = _stage(c1, cfg.enc_chains[0], gs)
        self.down2 = HaarConv1x1(c1, c2)
        self.stage2 = _stage(c2, cfg.enc_chains[1], gs)
        self.down3 = HaarConv1x1(c2, c3)
        self.stage3 = _stage(c3, cfg.enc_chains[2], gs)
        self.out = HaarConv1x1(c3, cfg.latent_channels)

        # Hyper-encoder absorbed into the encoder (Sec. 3.1).  Its architecture is
        # not given in the paper; we mirror the scale decoder of Fig. 11.
        w = cfg.side_width
        self.hyper = nn.Sequential(
            ConvScale(cfg.latent_channels, w, 1),
            CSChain(w, 2, 4, 1, gs),
            HaarConv1x1(w, w),
            CSChain(w, 2, 4, 1, gs),
            HaarConv1x1(w, w),
            CSChain(w, 2, 4, 1, gs),
            ConvScale(w, cfg.hyper_channels, 1),
        )

    def forward(self, x: torch.Tensor, lw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.inp(concat_level(x - 0.5, lw))
        h = self.stage1(h)
        h = self.stage2(self.down2(h))
        h = self.stage3(self.down3(h))
        y = self.out(h)
        return y, self.hyper(y)


# --------------------------------------------------------------------------- #
# Outer decoder (Fig. 4)
# --------------------------------------------------------------------------- #
class OuterDecoder(nn.Module):
    def __init__(self, cfg: PICOConfig):
        super().__init__()
        c1, c2, c3 = cfg.dec_channels
        gs = cfg.group_size
        self.up1 = Conv1x1iHaar(cfg.latent_channels + NUM_COARSE_LEVELS, c1)  # ConcatLevel
        self.stage1 = _stage(c1, cfg.dec_chains[0], gs)
        self.up2 = Conv1x1iHaar(c1, c2)
        self.stage2 = _stage(c2, cfg.dec_chains[1], gs)
        self.up3 = Conv1x1iHaar(c2, c3)
        self.stage3 = _stage(c3, cfg.dec_chains[2], gs)
        self.out = Conv1x1iHaar(c3, 3)

    def forward(self, y_hat: torch.Tensor, lw: torch.Tensor) -> torch.Tensor:
        h = self.stage1(self.up1(concat_level(y_hat, lw)))
        h = self.stage2(self.up2(h))
        h = self.stage3(self.up3(h))
        return self.out(h) + 0.5


# --------------------------------------------------------------------------- #
# Hyper decoders (Fig. 11, right)
# --------------------------------------------------------------------------- #
class _HyperDecoder(nn.Sequential):
    def __init__(self, cfg: PICOConfig, out_channels: int):
        w, gs = cfg.side_width, cfg.group_size
        super().__init__(
            ConvScale(cfg.hyper_channels, w, 1),
            CSChain(w, 2, 4, 1, gs),
            Conv1x1iHaar(w, w),
            CSChain(w, 2, 4, 1, gs),
            Conv1x1iHaar(w, w),
            CSChain(w, 2, 4, 1, gs),
            ConvScale(w, out_channels, 1),
        )


class ScaleDecoder(_HyperDecoder):
    """y^_hyper -> sigma.  Must be bit-exact between encoder and decoder."""

    def __init__(self, cfg: PICOConfig):
        super().__init__(cfg, cfg.latent_channels)

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        return F.softplus(super().forward(z_hat))


class ContextDecoder(_HyperDecoder):
    """y^_hyper -> prior p for the one-shot context model (same topology as the
    scale decoder; the paper does not give its exact architecture)."""

    def __init__(self, cfg: PICOConfig):
        super().__init__(cfg, cfg.context_channels)


# --------------------------------------------------------------------------- #
# One-shot context model (Sec. 4.1, Fig. 3 top-right)
# --------------------------------------------------------------------------- #
def ste_round(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()


def grid2x2_masks(h: int, w: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Chunkify masks of the 2x2-grid AR order, [4, 1, h, w].

    Step order within each 2x2 block: (0,0), (1,1), (0,1), (1,0).
    """
    order = [(0, 0), (1, 1), (0, 1), (1, 0)]
    masks = torch.zeros(4, 1, h, w, device=device, dtype=dtype)
    for n, (i, j) in enumerate(order):
        masks[n, :, i::2, j::2] = 1
    return masks


class Predictor(nn.Module):
    """Predictor_n: (p, previously decoded y^_1..n-1) -> (mu_n, q_n)."""

    def __init__(self, in_channels: int, width: int, latent_channels: int,
                 group_size: int, q_min: float):
        super().__init__()
        self.q_min = q_min
        self.net = nn.Sequential(
            ConvScale(in_channels, width, 1),
            CSChain(width, 2, 2, 1, group_size),
            ConvScale(width, 2 * latent_channels, 1),
        )
        # init: mu ~ 0, q ~ 1
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias[latent_channels:].fill_(_softplus_inv(1.0 - q_min))

    def forward(self, inp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, q = self.net(inp).chunk(2, dim=1)
        return mu, self.q_min + F.softplus(q)


def _softplus_inv(v: float) -> float:
    return float(torch.log(torch.expm1(torch.tensor(v))))


class OneShotContext(nn.Module):
    """Auto-regressive prediction of location mu and quantization width q in N
    steps (2x2 grid), while the entropy-coding scale sigma is predicted in one
    shot by the scale decoder.  Quantization: y^ = round((y - mu) / q);
    dequantization: q * y^ + mu.
    """

    num_steps = 4

    def __init__(self, cfg: PICOConfig):
        super().__init__()
        c, cp = cfg.latent_channels, cfg.context_channels
        self.predictors = nn.ModuleList([
            Predictor(cp if n == 0 else cp + c + 1, cfg.predictor_width, c,
                      cfg.group_size, cfg.q_min)
            for n in range(self.num_steps)
        ])

    def _inputs(self, n, p, y_rec, decoded):
        return p if n == 0 else torch.cat([p, y_rec, decoded.expand(p.shape[0], -1, -1, -1)], 1)

    def forward_train(self, y: torch.Tensor, p: torch.Tensor, noise: bool = True):
        """Returns (dequantized y [STE], noisy symbols for the rate term)."""
        masks = grid2x2_masks(y.shape[2], y.shape[3], y.device, y.dtype)
        y_rec = torch.zeros_like(y)
        sym_rate = torch.zeros_like(y)
        decoded = torch.zeros_like(masks[0:1])
        for n, pred in enumerate(self.predictors):
            m = masks[n:n + 1]
            mu, q = pred(self._inputs(n, p, y_rec, decoded))
            s = (y - mu) / q
            s_rate = s + torch.empty_like(s).uniform_(-0.5, 0.5) if noise else torch.round(s)
            y_rec = y_rec + m * (q * ste_round(s) + mu)
            sym_rate = sym_rate + m * s_rate
            decoded = decoded + m
        return y_rec, sym_rate

    @torch.no_grad()
    def encode(self, y: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (integer-valued symbols, dequantized latent)."""
        masks = grid2x2_masks(y.shape[2], y.shape[3], y.device, y.dtype)
        y_rec = torch.zeros_like(y)
        sym = torch.zeros_like(y)
        decoded = torch.zeros_like(masks[0:1])
        for n, pred in enumerate(self.predictors):
            m = masks[n:n + 1]
            mu, q = pred(self._inputs(n, p, y_rec, decoded))
            s = torch.round((y - mu) / q)
            sym = sym + m * s
            y_rec = y_rec + m * (q * s + mu)
            decoded = decoded + m
        return sym, y_rec

    @torch.no_grad()
    def decode(self, sym: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        masks = grid2x2_masks(sym.shape[2], sym.shape[3], sym.device, p.dtype)
        y_rec = torch.zeros(sym.shape, device=p.device, dtype=p.dtype)
        decoded = torch.zeros_like(masks[0:1])
        for n, pred in enumerate(self.predictors):
            m = masks[n:n + 1]
            mu, q = pred(self._inputs(n, p, y_rec, decoded))
            y_rec = y_rec + m * (q * sym + mu)
            decoded = decoded + m
        return y_rec


# --------------------------------------------------------------------------- #
# Full codec
# --------------------------------------------------------------------------- #
def default_scale_table(lo: float = 0.11, hi: float = 256.0, n: int = 64) -> List[float]:
    import math
    return torch.exp(torch.linspace(math.log(lo), math.log(hi), n)).tolist()


class PICO(nn.Module):
    def __init__(self, cfg: Optional[PICOConfig] = None):
        super().__init__()
        self.cfg = cfg = cfg or PICOConfig()
        self.encoder = OuterEncoder(cfg)
        self.decoder = OuterDecoder(cfg)
        self.scale_decoder = ScaleDecoder(cfg)
        self.context_decoder = ContextDecoder(cfg)
        self.context = OneShotContext(cfg)
        self.entropy_bottleneck = EntropyBottleneck(cfg.hyper_channels)
        self.gaussian_conditional = GaussianConditional(None, scale_bound=cfg.scale_bound)
        # Learned level-conditional channel-wise gain on the latent (App. F),
        # stored in log-space, interpolated across levels.
        self.log_gain = nn.Parameter(torch.zeros(NUM_COARSE_LEVELS, cfg.latent_channels))

    # ---- helpers ---------------------------------------------------------- #
    def gain(self, lw: torch.Tensor) -> torch.Tensor:
        return torch.exp(lw @ self.log_gain)[:, :, None, None]

    def _z_hat(self, z: torch.Tensor) -> torch.Tensor:
        medians = self.entropy_bottleneck._get_medians().detach().view(1, -1, 1, 1)
        return ste_round(z - medians) + medians

    def aux_loss(self) -> torch.Tensor:
        return self.entropy_bottleneck.loss()

    # ---- training forward ------------------------------------------------- #
    def forward(self, x: torch.Tensor, level: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x in [0, 1], [B, 3, H, W] with H, W multiples of 64; level [B] in [0, 7]."""
        lw = level_weights(level)
        y, z = self.encoder(x, lw)
        _, z_lik = self.entropy_bottleneck(z)
        z_hat = self._z_hat(z) if self.training else self.entropy_bottleneck.quantize(
            z, "dequantize", self.entropy_bottleneck._get_medians().view(1, -1, 1, 1))
        sigma = self.scale_decoder(z_hat)
        p = self.context_decoder(z_hat)
        g = self.gain(lw)
        y_rec, sym = self.context.forward_train(y * g, p, noise=self.training)
        with torch.autocast(device_type=x.device.type, enabled=False):
            y_lik = self.gaussian_conditional._likelihood(sym.float(), sigma.float())
            y_lik = self.gaussian_conditional.likelihood_lower_bound(y_lik)
        x_hat = self.decoder(y_rec / g, lw)
        return {"x_hat": x_hat, "likelihoods": {"y": y_lik, "z": z_lik}, "y": y}

    # ---- entropy coding --------------------------------------------------- #
    def update(self, scale_table: Optional[Sequence[float]] = None, force: bool = True) -> bool:
        """Build the CDF tables used by the range coder.  Call before compress()."""
        scale_table = list(scale_table or default_scale_table(self.cfg.scale_bound))
        a = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        b = self.entropy_bottleneck.update(force=force)
        return a or b

    def load_state_dict(self, state_dict, strict: bool = True):
        update_registered_buffers(
            self.entropy_bottleneck, "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"], state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional, "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"], state_dict,
        )
        return super().load_state_dict(state_dict, strict=strict)

    @torch.no_grad()
    def deterministic_sigma(self, z_hat: torch.Tensor) -> torch.Tensor:
        """Scale decoder on CPU (IEEE FP) -> sigma.  Section 3.2 runs this on CPU
        so that encoder and decoder produce identical CDF indexes (call
        ``to_inference`` to place it there)."""
        dev = next(self.scale_decoder.parameters()).device
        return self.scale_decoder(z_hat.detach().to(dev, torch.float32)).cpu()

    @torch.no_grad()
    def compress(self, x: torch.Tensor, level: float) -> Dict:
        """Encode a batch of tiles at one quality level.

        The scale decoder must live on the CPU (see ``PICO.to_inference``)."""
        self._entropy_parts_to_cpu()
        dev = x.device
        lw = level_weights(torch.full((x.shape[0],), float(level), device=dev))
        y, z = self.encoder(x, lw)
        z_strings = self.entropy_bottleneck.compress(z.cpu())
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.shape[-2:])
        sigma = self.deterministic_sigma(z_hat)
        indexes = self.gaussian_conditional.build_indexes(sigma)
        p = self.context_decoder(z_hat.to(dev))
        sym, _ = self.context.encode(y * self.gain(lw), p)
        y_strings = self.gaussian_conditional.compress(sym.cpu(), indexes)
        return {"strings": [y_strings, z_strings], "shape": tuple(z.shape[-2:])}

    @torch.no_grad()
    def decompress(self, strings, shape, level: float, device=None) -> torch.Tensor:
        self._entropy_parts_to_cpu()
        y_strings, z_strings = strings
        dev = device or next(self.decoder.parameters()).device
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        sigma = self.deterministic_sigma(z_hat)
        indexes = self.gaussian_conditional.build_indexes(sigma)
        sym = self.gaussian_conditional.decompress(y_strings, indexes)
        lw = level_weights(torch.full((len(y_strings),), float(level), device=dev))
        p = self.context_decoder(z_hat.to(dev))
        y_rec = self.context.decode(sym.to(dev), p)
        return self.decoder(y_rec / self.gain(lw), lw).clamp(0, 1)

    def _entropy_parts_to_cpu(self) -> None:
        for m in (self.scale_decoder, self.entropy_bottleneck, self.gaussian_conditional):
            m.to("cpu")

    # ---- deployment ------------------------------------------------------- #
    def to_inference(self, device="cpu", reparameterize: bool = True) -> "PICO":
        """eval(), optionally collapse ConvScale/Haar into plain convs, move the
        neural transforms to `device` and keep the scale decoder + entropy models
        on the CPU."""
        self.eval()
        if reparameterize:
            for m in (self.encoder, self.decoder, self.context_decoder, self.context):
                reparameterize_module(m)
        self.to(device)
        self._entropy_parts_to_cpu()
        return self


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
