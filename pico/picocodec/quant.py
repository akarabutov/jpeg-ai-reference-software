"""Integer-only scale decoder for bit-exact, cross-platform entropy coding.

Sec. 3.2 of the paper: the parameters given to the entropy coder must be
bit-exact between encoder and decoder, so the scale decoder is quantized to
UINT8 and run in a deterministic way.  Here we go one step further and remove
floating point from the runtime path altogether:

* weights:      INT8, symmetric, per output channel
* activations:  UINT8, asymmetric (scale, zero point), per tensor
* conv:         integer accumulation  acc = sum w_q * (x_q - zp_x) + b_q
* requantize:   y_q = clamp(zp_y + round(acc * M0 / 2^S), 0, 255) with integer
                M0, S per output channel (gemmlowp / TFLite style)
* GELU:         256-entry UINT8 look-up table
* residual add: fixed-point rescale of both inputs
* pixel shuffle (inverse Haar is folded into the conv weights): pure reorder
* softplus + CDF index:  integer thresholds per (channel, scale-table entry),
                index = #thresholds the accumulator exceeds -- this is exactly
                ``GaussianConditional.build_indexes(softplus(.))`` in exact
                arithmetic.

Convolutions are executed as float64 convolutions on integer-valued tensors.
All products and partial sums are integers far below 2^53, so every IEEE-754
float64 implementation returns the exact integer result regardless of the
summation order, threading, vectorisation or FMA usage; the kernels can be
ported 1:1 to int32 GEMMs.  Everything else is int64 arithmetic.

Calibration is done layer by layer: each layer's weight scales, (optionally)
bias correction and output activation range are fitted on the activations
produced by the *already quantized* prefix of the network, so that the
quantization error of earlier layers is taken into account.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Conv1x1iHaar, ConvScale, CSChain

QMIN, QMAX = 0, 255          # activations (uint8)
WMAX = 127                    # weights (int8, symmetric)
ADD_SHIFT = 24                # fixed-point precision of the residual add
ACC_LIMIT = 2 ** 31           # accumulators must fit int32 (exactness / no overflow)
SYM_LIMIT = 2 ** 15           # hyper-latent symbols are clamped to +-SYM_LIMIT
FORMAT_VERSION = 1


# --------------------------------------------------------------------------- #
# Float graph of the scale decoder
# --------------------------------------------------------------------------- #
def _conv_spec(m: nn.Module) -> Dict:
    if isinstance(m, ConvScale):
        w = m.effective_weight().detach().double()
        b = m.effective_bias()
        b = b.detach().double() if b is not None else torch.zeros(w.shape[0], dtype=torch.float64)
        return {"w": w, "b": b, "groups": m.groups, "pad": m.padding}
    if isinstance(m, nn.Conv2d):
        assert m.stride == (1, 1) and m.dilation == (1, 1)
        b = m.bias.detach().double() if m.bias is not None else torch.zeros(
            m.out_channels, dtype=torch.float64)
        return {"w": m.weight.detach().double(), "b": b, "groups": m.groups,
                "pad": m.padding[0]}
    raise TypeError(f"not a convolution: {type(m)}")


def float_ops(scale_decoder: nn.Module) -> List[Dict]:
    """Flatten a (possibly reparameterised) ScaleDecoder into a list of ops:
    conv / block (ConvScale311) / shuffle / out (last conv, followed by softplus)."""
    ops: List[Dict] = []
    for m in scale_decoder.children():
        if isinstance(m, (ConvScale, nn.Conv2d)):
            ops.append({"type": "conv", **_conv_spec(m)})
        elif isinstance(m, CSChain):
            for blk in m:
                ops.append({"type": "block", "convs": [
                    _conv_spec(blk.conv3), _conv_spec(blk.conv1a), _conv_spec(blk.conv1b)]})
        elif isinstance(m, Conv1x1iHaar):
            rep = m.reparameterize()
            ops += [{"type": "conv", **_conv_spec(rep[0])}, {"type": "shuffle", "factor": 2}]
        elif isinstance(m, nn.Sequential) and len(m) == 2 and isinstance(m[1], nn.PixelShuffle):
            ops += [{"type": "conv", **_conv_spec(m[0])},
                    {"type": "shuffle", "factor": m[1].upscale_factor}]
        else:
            raise TypeError(f"unsupported layer in scale decoder: {type(m)}")
    assert ops and ops[-1]["type"] == "conv", "scale decoder must end with a conv"
    ops[-1]["type"] = "out"
    assert ops[0]["type"] == "conv" and ops[0]["w"].shape[-1] == 1 and ops[0]["groups"] == 1, \
        "first layer must be a dense 1x1 conv (medians are folded into its bias)"
    return ops


def _fconv(x: torch.Tensor, c: Dict, w: Optional[torch.Tensor] = None,
           b: Optional[torch.Tensor] = None) -> torch.Tensor:
    return F.conv2d(x, c["w"] if w is None else w, c["b"] if b is None else b,
                    padding=c["pad"], groups=c["groups"])


def float_forward(ops: List[Dict], z_hat: torch.Tensor) -> torch.Tensor:
    """Reference float64 execution of the op list -> sigma."""
    x = z_hat.double()
    for op in ops:
        t = op["type"]
        if t in ("conv", "out"):
            x = _fconv(x, op)
        elif t == "block":
            c3, c1a, c1b = op["convs"]
            x = x + _fconv(F.gelu(_fconv(F.gelu(_fconv(x, c3)), c1a)), c1b)
        elif t == "shuffle":
            x = F.pixel_shuffle(x, op["factor"])
    return F.softplus(x)


# --------------------------------------------------------------------------- #
# Integer primitives (used both at calibration time and at runtime)
# --------------------------------------------------------------------------- #
def int_conv(xq: torch.Tensor, zp: int, wq: torch.Tensor, bq: torch.Tensor,
             groups: int, pad: int) -> torch.Tensor:
    """Exact integer convolution, int64 in / int64 out."""
    x = (xq - zp).to(torch.float64)
    acc = F.conv2d(x, wq.to(torch.float64), None, padding=pad, groups=groups)
    acc = torch.round(acc).to(torch.int64) + bq.view(1, -1, 1, 1)
    if acc.numel() and acc.abs().max().item() >= ACC_LIMIT:
        raise OverflowError("accumulator exceeds int32 range")
    return acc


def rshift_round(v: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """round(v / 2^shift) (half up) with integer ops; shift >= 1."""
    one = torch.ones_like(shift)
    return torch.bitwise_right_shift(v + torch.bitwise_left_shift(one, shift - 1), shift)


def requantize(acc: torch.Tensor, m0: torch.Tensor, shift: torch.Tensor, zp: int) -> torch.Tensor:
    v = rshift_round(acc * m0.view(1, -1, 1, 1), shift.view(1, -1, 1, 1))
    return (v + zp).clamp(QMIN, QMAX)


def int_add(xq, zx, mx, bq, zb, mb, zo) -> torch.Tensor:
    s = torch.tensor(ADD_SHIFT, dtype=torch.int64)
    v = rshift_round((xq - zx) * mx + (bq - zb) * mb, s)
    return (v + zo).clamp(QMIN, QMAX)


def fixed_point(m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Positive float multipliers -> (M0, S) with m ~= M0 / 2^S, M0 in [2^30, 2^31)."""
    m0s, ss = [], []
    for v in m.double().tolist():
        if not v > 0:
            raise ValueError("multiplier must be positive")
        e = math.floor(math.log2(v))
        s = 30 - e
        q = round(v * 2.0 ** s)
        if q >= 2 ** 31:
            q //= 2
            s -= 1
        if not 1 <= s <= 62:
            raise ValueError(f"multiplier {v} out of the supported range")
        m0s.append(q)
        ss.append(s)
    return torch.tensor(m0s, dtype=torch.int64), torch.tensor(ss, dtype=torch.int64)


# --------------------------------------------------------------------------- #
# Calibration helpers
# --------------------------------------------------------------------------- #
def _range(t: torch.Tensor, percentile: float, max_samples: int = 2_000_000) -> Tuple[float, float]:
    flat = t.detach().flatten().double()
    if flat.numel() > max_samples:
        g = torch.Generator().manual_seed(0)
        flat = flat[torch.randint(0, flat.numel(), (max_samples,), generator=g)]
    if percentile >= 1.0:
        return flat.min().item(), flat.max().item()
    return (torch.quantile(flat, 1 - percentile).item(), torch.quantile(flat, percentile).item())


def act_qparams(t: torch.Tensor, percentile: float) -> Tuple[float, int]:
    lo, hi = _range(t, percentile)
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi - lo < 1e-8:
        hi = lo + 1e-8
    s = (hi - lo) / (QMAX - QMIN)
    zp = int(min(QMAX, max(QMIN, round(-lo / s))))
    return s, zp


def quantize_act(t: torch.Tensor, s: float, zp: int) -> torch.Tensor:
    return (torch.round(t / s) + zp).clamp(QMIN, QMAX).to(torch.int64)


def sqnr_db(ref: torch.Tensor, approx: torch.Tensor) -> float:
    noise = (ref - approx).pow(2).sum().item()
    return float("inf") if noise == 0 else 10 * math.log10(ref.pow(2).sum().item() / noise)


def softplus_inv(t: torch.Tensor) -> torch.Tensor:
    t = t.double()
    return t + torch.log(-torch.expm1(-t))


# --------------------------------------------------------------------------- #
# Quantized scale decoder
# --------------------------------------------------------------------------- #
class QuantScaleDecoder:
    """Integer scale decoder: hyper-latent symbols k (z_hat = k + medians) ->
    Gaussian-conditional CDF indexes.  Plain object (not an nn.Module) so that
    it never ends up in the float model's state dict."""

    def __init__(self, ops: List[Dict], scale_table: Sequence[float]):
        self.ops = ops
        self.scale_table = [float(s) for s in scale_table]

    # ---- runtime ----------------------------------------------------------- #
    @torch.no_grad()
    def __call__(self, k: torch.Tensor) -> torch.Tensor:
        """k: integer hyper-latent symbols [B, C, h, w] -> int32 indexes."""
        xq = k.to("cpu", torch.int64).clamp(-SYM_LIMIT, SYM_LIMIT)
        zp = 0
        for op in self.ops:
            t = op["type"]
            if t == "conv":
                acc = int_conv(xq, zp, op["wq"], op["bq"], op["groups"], op["pad"])
                xq, zp = requantize(acc, op["m0"], op["shift"], op["zp_out"]), op["zp_out"]
            elif t == "block":
                xin, zin = xq, zp
                h = xq
                hz = zp
                for i, c in enumerate(op["convs"]):
                    acc = int_conv(h, hz, c["wq"], c["bq"], c["groups"], c["pad"])
                    h, hz = requantize(acc, c["m0"], c["shift"], c["zp_out"]), c["zp_out"]
                    if i < 2:  # GELU
                        h, hz = op["luts"][i][h], op["lut_zp"][i]
                a = op["add"]
                xq = int_add(xin, zin, a["mx"], h, hz, a["mb"], a["zp_out"])
                zp = a["zp_out"]
            elif t == "shuffle":
                xq = F.pixel_shuffle(xq, op["factor"])
            elif t == "out":
                acc = int_conv(xq, zp, op["wq"], op["bq"], op["groups"], op["pad"])
                return self._indexes(acc, op["thresholds"])
        raise RuntimeError("missing output layer")

    def _indexes(self, acc: torch.Tensor, thr: torch.Tensor) -> torch.Tensor:
        # index = (L-1) - #{i < L-1 : sigma <= t_i};  sigma <= t_i  <=>  acc <= thr[c, i]
        n = len(self.scale_table) - 1
        le = torch.zeros_like(acc)
        for i in range(n):
            le += (acc <= thr[:, i].view(1, -1, 1, 1)).to(torch.int64)
        return (n - le).to(torch.int32)

    # ---- (de)serialisation -------------------------------------------------- #
    def to_dict(self) -> Dict:
        return {"format_version": FORMAT_VERSION, "ops": self.ops, "scale_table": self.scale_table}

    @classmethod
    def from_dict(cls, d: Dict) -> "QuantScaleDecoder":
        if d.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported quantized scale decoder format {d.get('format_version')}")
        return cls(d["ops"], d["scale_table"])

    def num_bytes(self) -> int:
        """Size of the int8 weights (what would be shipped to a device)."""
        n = 0
        for op in self.ops:
            for c in op.get("convs", [op] if "wq" in op else []):
                n += c["wq"].numel()
        return n

    # ---- calibration -------------------------------------------------------- #
    @classmethod
    @torch.no_grad()
    def calibrate(
        cls,
        scale_decoder: nn.Module,
        symbols: torch.Tensor,
        medians: torch.Tensor,
        scale_table: Sequence[float],
        percentile: float = 0.9999,
        bias_correction: bool = True,
    ) -> Tuple["QuantScaleDecoder", List[Dict]]:
        """Layer-by-layer post-training quantization.

        symbols: calibration hyper-latent symbols k [N, C, h, w] (integers);
        medians: entropy-bottleneck medians [C]; z_hat = k + medians.
        Returns the quantized decoder and a per-layer report.
        """
        fops = float_ops(scale_decoder)
        med = medians.detach().double().view(1, -1, 1, 1).cpu()
        k = symbols.detach().cpu().to(torch.int64).clamp(-SYM_LIMIT, SYM_LIMIT)
        ref = k.double() + med                     # float reference activations
        xq, s, zp = k, 1.0, 0                      # quantized path: real = (xq - zp) * s
        qops: List[Dict] = []
        report: List[Dict] = []
        table = torch.tensor(scale_table, dtype=torch.float64)

        def qconv(c: Dict, xq, s, zp, ref, name: str, fold_medians: bool = False):
            """Quantize one conv: weights, (corrected) bias.  Returns integer
            accumulator, its per-channel scale and the float reference output."""
            w, b = c["w"], c["b"]
            y_ref = _fconv(ref, c)
            if fold_medians:  # z_hat = k + m  ->  conv(z_hat) = conv(k) + W m
                b = b + c["w"][:, :, 0, 0] @ med.view(-1)
            s_w = w.abs().amax(dim=(1, 2, 3)).clamp_min(1e-12) / WMAX
            wq = torch.round(w / s_w.view(-1, 1, 1, 1)).clamp(-WMAX, WMAX)
            x_deq = (xq - zp).double() * s
            if bias_correction:
                y_q = _fconv(x_deq, c, w=wq * s_w.view(-1, 1, 1, 1), b=b)
                b = b - (y_q - y_ref).mean(dim=(0, 2, 3))
            s_acc = s * s_w
            bq = torch.round(b / s_acc).to(torch.int64)
            acc = int_conv(xq, zp, wq.to(torch.int64), bq, c["groups"], c["pad"])
            spec = {"wq": wq.to(torch.int8), "bq": bq, "groups": c["groups"], "pad": c["pad"]}
            return acc, s_acc, y_ref, spec

        def requant_spec(spec, acc, s_acc, y_ref, name):
            """Choose output range of a conv (on the quantized path), requantize."""
            y = acc.double() * s_acc.view(1, -1, 1, 1)
            s_o, zp_o = act_qparams(y, percentile)
            m0, sh = fixed_point(s_acc / s_o)
            spec.update(m0=m0, shift=sh, zp_out=zp_o)
            out = requantize(acc, m0, sh, zp_o)
            report.append({"layer": name, "shape": tuple(out.shape[1:]), "scale": s_o, "zp": zp_o,
                           "sqnr_db": sqnr_db(y_ref, (out - zp_o).double() * s_o)})
            return out, s_o, zp_o

        def gelu_lut(hq, s_i, zp_i, ref_pre, name):
            levels = (torch.arange(QMIN, QMAX + 1, dtype=torch.float64) - zp_i) * s_i
            g_vals = F.gelu((hq - zp_i).double() * s_i)
            s_o, zp_o = act_qparams(g_vals, 1.0)
            lut = quantize_act(F.gelu(levels), s_o, zp_o)
            out = lut[hq]
            g_ref = F.gelu(ref_pre)
            report.append({"layer": name, "shape": tuple(out.shape[1:]), "scale": s_o, "zp": zp_o,
                           "sqnr_db": sqnr_db(g_ref, (out - zp_o).double() * s_o)})
            return out, s_o, zp_o, lut, g_ref

        n_conv = n_block = n_up = 0
        for i, op in enumerate(fops):
            t = op["type"]
            if t == "conv":
                name = f"conv{n_conv}"
                n_conv += 1
                acc, s_acc, y_ref, spec = qconv(op, xq, s, zp, ref, name, fold_medians=(i == 0))
                xq, s, zp = requant_spec(spec, acc, s_acc, y_ref, name)
                ref = y_ref
                qops.append({"type": "conv", **spec})
            elif t == "block":
                name = f"block{n_block}"
                n_block += 1
                xin, s_in, zp_in, ref_in = xq, s, zp, ref
                h, hs, hz, href = xq, s, zp, ref
                specs, luts, lut_zp = [], [], []
                for j, c in enumerate(op["convs"]):
                    acc, s_acc, y_ref, spec = qconv(c, h, hs, hz, href, f"{name}.conv{j}")
                    h, hs, hz = requant_spec(spec, acc, s_acc, y_ref, f"{name}.conv{j}")
                    href = y_ref
                    specs.append(spec)
                    if j < 2:
                        h, hs, hz, lut, href = gelu_lut(h, hs, hz, href, f"{name}.gelu{j}")
                        luts.append(lut)
                        lut_zp.append(hz)
                real = (xin - zp_in).double() * s_in + (h - hz).double() * hs
                s_o, zp_o = act_qparams(real, percentile)
                mx = torch.tensor(round(s_in / s_o * 2 ** ADD_SHIFT), dtype=torch.int64)
                mb = torch.tensor(round(hs / s_o * 2 ** ADD_SHIFT), dtype=torch.int64)
                xq = int_add(xin, zp_in, mx, h, hz, mb, zp_o)
                s, zp = s_o, zp_o
                ref = ref_in + href
                report.append({"layer": f"{name}.add", "shape": tuple(xq.shape[1:]), "scale": s,
                               "zp": zp, "sqnr_db": sqnr_db(ref, (xq - zp).double() * s)})
                qops.append({"type": "block", "convs": specs, "luts": luts, "lut_zp": lut_zp,
                             "add": {"mx": mx, "mb": mb, "zp_out": zp_o}})
            elif t == "shuffle":
                n_up += 1
                xq = F.pixel_shuffle(xq, op["factor"])
                ref = F.pixel_shuffle(ref, op["factor"])
                qops.append({"type": "shuffle", "factor": op["factor"]})
            elif t == "out":
                acc, s_acc, y_ref, spec = qconv(op, xq, s, zp, ref, "out")
                # sigma <= t  <=>  acc * s_acc <= softplus^-1(t)  <=>  acc <= floor(.)
                thr = torch.floor(softplus_inv(table[:-1]).view(1, -1) / s_acc.view(-1, 1))
                thr = thr.clamp(-2 ** 62, 2 ** 62).to(torch.int64)
                spec["thresholds"] = thr
                qops.append({"type": "out", **spec})
                sig_q = F.softplus(acc.double() * s_acc.view(1, -1, 1, 1))
                report.append({"layer": "out(sigma)", "shape": tuple(acc.shape[1:]), "scale": None,
                               "zp": None, "sqnr_db": sqnr_db(F.softplus(y_ref), sig_q)})
        return cls(qops, scale_table), report


def float_indexes(sigma: torch.Tensor, scale_table: Sequence[float], scale_bound: float = 0.11):
    """CompressAI ``build_indexes`` in float64 (reference for comparisons)."""
    table = torch.tensor(scale_table, dtype=torch.float64)
    # same float32 lower bound as CompressAI's LowerBound(scale_bound)
    bound = torch.tensor(scale_bound, dtype=torch.float32).double()
    s = sigma.double().clamp_min(bound)
    idx = torch.full(s.shape, len(table) - 1, dtype=torch.int64)
    for t in table[:-1]:
        idx -= (s <= t).to(torch.int64)
    return idx.to(torch.int32)
