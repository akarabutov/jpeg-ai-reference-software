"""PyTorch Lightning wrapper for PICO training.

Manual optimisation is used because one training iteration contains up to
three coupled updates (codec, entropy-bottleneck quantiles, discriminator):

  1. codec step     -- rate-distortion loss (+ adversarial term in phase 2)
                       and the auxiliary quantile loss of the factorized entropy
                       model; quantiles are a separate param group of the same
                       Adam with their own lr and no gradient clipping.
  2. discriminator  -- (phase 2 only) on real images and detached
                       reconstructions, optional lazy R1 penalty.

Lightning's ``global_step`` counts optimizer steps (2 per iteration with a
discriminator), so ``self.iteration`` is used for schedules; logged curves use
the batch counter, i.e. one point per iteration.
"""

from __future__ import annotations

from typing import Dict

import lightning as L
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from .discriminator import PatchDiscriminator, discriminator_loss, generator_loss, r1_penalty
from .layers import NUM_FINE_LEVELS, fine_index_to_level
from .losses import LossConfig, RateDistortionLoss, bits_per_pixel
from .model import PICO, PICOConfig


def lr_lambda_factory(total: int, milestones, factors):
    """Piecewise-constant multiplier: `factors[i]` from `milestones[i]` (a
    fraction of `total` iterations) on."""
    def fn(it: int) -> float:
        frac = it / max(1, total)
        f = 1.0
        for m, fac in zip(milestones, factors):
            if frac >= m:
                f = fac
        return f
    return fn


def extract_pico_state_dict(ck: dict) -> dict:
    """Plain PICO state dict from either a Lightning checkpoint or an exported one."""
    if "model" in ck:
        return ck["model"]
    sd = ck["state_dict"]
    return {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}


class PICOLightning(L.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters({"config": config})
        self.config = config
        self.model_cfg = PICOConfig.from_dict(config.get("model"))
        self.model = PICO(self.model_cfg)
        self.criterion = RateDistortionLoss(LossConfig.from_dict(config["loss"]))
        self.phase = self.criterion.cfg.phase

        dcfg = config.get("disc", {}) or {}
        self.use_gan = self.phase == "perceptual" and self.criterion.cfg.w_gan > 0
        self.gan_loss_type = dcfg.get("loss", "ns")
        self.r1_gamma = float(dcfg.get("r1_gamma", 0.0))
        self.r1_every = int(dcfg.get("r1_every", 16))
        self.disc = PatchDiscriminator(**dcfg.get("arch", {})) if self.use_gan else None

        data = config.get("data", {})
        self.use_text = self.phase == "perceptual" and bool(data.get("mask_root"))
        tr = config["train"]
        self.total_iters = int(tr["steps"])
        self.val_levels = [float(v) for v in tr.get("val_levels", [0.0, 3.5, 7.0])]
        self.val_images = int(tr.get("val_log_images", 4))
        self.grad_clip = float(tr.get("grad_clip", 1.0) or 0.0)
        self.automatic_optimization = False

    # ------------------------------------------------------------------ #
    @property
    def opt_steps_per_iter(self) -> int:
        return 2 if self.use_gan else 1

    @property
    def iteration(self) -> int:
        return self.trainer.global_step // self.opt_steps_per_iter

    def load_pico_weights(self, path: str) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(extract_pico_state_dict(ck))

    # ------------------------------------------------------------------ #
    def configure_optimizers(self):
        tr = self.config["train"]
        main = [p for n, p in self.model.named_parameters() if not n.endswith(".quantiles")]
        self._main_params = main
        g_opt = torch.optim.Adam([
            {"params": main, "lr": tr["lr"]},
            {"params": [self.model.entropy_bottleneck.quantiles], "lr": tr.get("aux_lr", 1e-3)},
        ])
        fn = lr_lambda_factory(self.total_iters, tr["lr_milestones"], tr["lr_factors"])
        opts, scheds = [g_opt], [LambdaLR(g_opt, [fn, lambda _: 1.0])]
        if self.use_gan:
            dcfg = self.config.get("disc", {}) or {}
            d_opt = torch.optim.Adam(self.disc.parameters(), lr=dcfg.get("lr", 1e-4),
                                     betas=tuple(dcfg.get("betas", (0.5, 0.999))))
            opts.append(d_opt)
            scheds.append(LambdaLR(d_opt, fn))  # same piecewise decay as the codec
        return opts, scheds

    def _opts(self):
        o = self.optimizers()
        return o if isinstance(o, list) else [o]

    def _scheds(self):
        s = self.lr_schedulers()
        return s if isinstance(s, list) else [s]

    # ------------------------------------------------------------------ #
    def training_step(self, batch, batch_idx):
        it = self.iteration
        x = batch["x"]
        mask = batch["mask"] if self.use_text else None
        b = x.shape[0]
        level = fine_index_to_level(
            torch.randint(0, NUM_FINE_LEVELS, (b,), device=x.device).float())
        opts = self._opts()
        g_opt = opts[0]
        logs: Dict[str, torch.Tensor] = {}

        # ---- 1. codec (generator) step --------------------------------- #
        if self.use_gan:
            self.toggle_optimizer(g_opt)  # freezes the discriminator
        out = self.model(x, level)
        out["x_hat"] = out["x_hat"].float()
        g_gan = None
        w_gan = self.criterion.gan_weight(it) if self.use_gan else 0.0
        if self.use_gan and w_gan > 0:
            d_fake_g = self.disc(out["x_hat"]).float()
            g_gan = generator_loss(d_fake_g, mask, self.gan_loss_type)
        rd = self.criterion(out, x, level, mask, g_gan, it)
        aux = self.model.aux_loss()

        g_opt.zero_grad(set_to_none=True)
        self.manual_backward(rd["loss"] + aux)
        if self.grad_clip:
            logs["grad_norm"] = torch.nn.utils.clip_grad_norm_(self._main_params, self.grad_clip)
        g_opt.step()
        if self.use_gan:
            self.untoggle_optimizer(g_opt)
        logs.update({k: v.detach() for k, v in rd.items()})
        logs["aux_loss"] = aux.detach()

        # ---- 2. discriminator step ------------------------------------- #
        if self.use_gan:
            d_opt = opts[1]
            self.toggle_optimizer(d_opt)  # freezes the codec
            do_r1 = self.r1_gamma > 0 and it % self.r1_every == 0
            x_real = x.detach().requires_grad_(do_r1)
            d_real = self.disc(x_real).float()
            d_fake = self.disc(out["x_hat"].detach()).float()
            d_loss = discriminator_loss(d_real, d_fake, self.gan_loss_type)
            if do_r1:
                r1 = r1_penalty(d_real, x_real)
                # lazy regularisation: scale by the interval (StyleGAN2)
                d_loss = d_loss + 0.5 * self.r1_gamma * self.r1_every * r1
                logs["gan/r1"] = r1.detach()
            d_opt.zero_grad(set_to_none=True)
            self.manual_backward(d_loss)
            d_opt.step()
            self.untoggle_optimizer(d_opt)
            logs.update({
                "gan/d_loss": d_loss.detach(),
                "gan/d_real_logit": d_real.detach().mean(),
                "gan/d_fake_logit": d_fake.detach().mean(),
                "gan/d_acc": 0.5 * ((d_real > 0).float().mean() + (d_fake < 0).float().mean()).detach(),
                "gan/weight": torch.tensor(w_gan),
            })
            if "gan_g" in logs:
                logs["gan/g_loss"] = logs.pop("gan_g")

        logs["lr"] = torch.tensor(g_opt.param_groups[0]["lr"])  # lr used in this iteration
        for s in self._scheds():
            s.step()

        logs["iteration"] = torch.tensor(float(it + 1))
        self.log_dict({f"train/{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False,
                      batch_size=b)
        self.log("bpp", rd["bpp"], prog_bar=True, logger=False, batch_size=b)
        self.log("psnr", rd["psnr"], prog_bar=True, logger=False, batch_size=b)

    # ------------------------------------------------------------------ #
    def validation_step(self, batch, batch_idx):
        x = batch["x"]
        b, _, h, w = x.shape
        recons = []
        for lvl in self.val_levels:
            out = self.model(x, torch.full((b,), lvl, device=x.device))
            x_hat = out["x_hat"].float().clamp(0, 1)
            bpp = bits_per_pixel(out["likelihoods"], h * w).mean()
            mse = F.mse_loss(x_hat, x)
            psnr = 10 * torch.log10(1.0 / mse.clamp_min(1e-10))
            self.log_dict({f"val/q{lvl:g}/bpp": bpp, f"val/q{lvl:g}/psnr": psnr},
                          on_epoch=True, batch_size=b, sync_dist=True)
            recons.append(x_hat)
        if batch_idx < self.val_images:
            self._log_images(batch_idx, x, recons)

    def _log_images(self, idx, x, recons):
        exp = getattr(self.logger, "experiment", None)
        if exp is None or not hasattr(exp, "add_image") or self.global_rank != 0:
            return
        # original | reconstructions at val_levels (left to right)
        row = torch.cat([x[0]] + [r[0] for r in recons], dim=-1).clamp(0, 1)
        exp.add_image(f"val_images/{idx}", row.cpu(), global_step=self.iteration)

    # ------------------------------------------------------------------ #
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # Also store the plain codec so that pico_codec.py / evaluate.py can
        # load Lightning checkpoints directly.
        checkpoint["model"] = self.model.state_dict()
        checkpoint["model_cfg"] = self.model_cfg.to_dict()
        checkpoint["iteration"] = self.iteration
