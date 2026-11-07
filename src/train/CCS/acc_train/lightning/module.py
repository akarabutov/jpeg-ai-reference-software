# The copyright in this software is being made available under the BSD
# License, included below. This software may be subject to other third party
# and contributor rights, including patent rights, and no such rights are
# granted under this license.
#
# Copyright (c) 2010-2026, ITU/ISO/IEC
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
# be used to endorse or promote products derived from this software without
# specific prior written permission.
# 
# DISCLAIMER
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
"""LightningModule of the CCS training."""

import gc
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_

from src.train.CCS.acc_train.graph_utils.loss import set_criterion
from src.train.CCS.acc_train.graph_utils.lr_scheduler import set_lr_scheduler
from src.train.CCS.acc_train.graph_utils.optimizer import init_optimizer
from src.train.CCS.acc_train.utils.mp_env import set_random
from src.train.CCS.acc_train.utils.train_utils import (create_meters, has_nan, print_meters,
                                                       update_meters)

from .compat import pl, recreate_train_iterator_if_created

__all__ = ['CCSLightningModule']

BASE_BATCH_SIZE = 8  # the learning rate is scaled by (batch_size * world_size / BASE_BATCH_SIZE)
GRAD_CLIP_NORM = 10.0
AMP_MIN_SCALE = 2048.0
AMP_MAX_NAN_RATIO = 0.02  # AMP is switched off for the rest of the epoch above this ratio of NaN steps
MAX_EPOCHS_PER_RUN = 100
VAL_METER_KEYS = ['Rate', 'Hyper_rate', 'Y_PSNR', 'U_PSNR', 'V_PSNR', 'MSE', 'MSSSIM', 'Loss']
TRAIN_METER_KEYS = ['Time', 'rd_loss', 'reg_loss', 'lr', 'scale']


class CCSLightningModule(pl.LightningModule):
    """Training of ``model.Net`` with manual optimization.

    Manual optimization keeps the behaviour of the original training loop:

    * a random beta (from ``--beta_list``) per training step and a fixed cycle of betas in validation;
    * per-step learning-rate schedulers (``*.step``) are stepped before the forward pass, per-epoch
      schedulers (``*.epoch``) at the beginning of an epoch;
    * mixed precision with a ``GradScaler`` created per epoch: a NaN loss on any process re-runs the
      step in fp32, an overflow of the gradients (on any process) skips the AMP update and re-runs
      the step in fp32, the scale never goes below 2048, and AMP is switched off for the rest of the
      epoch when more than 2% of the steps failed;
    * gradients are clipped to the norm 10 per sub-network;
    * the random generators are re-seeded with ``seed + epoch`` at the beginning of each epoch.

    Args:
        net (Net): the model (not wrapped into DDP yet, the wrapping is done in :meth:`setup`).
        args (argparse.Namespace): arguments of ``utils.get_args``.
        legacy_resume (dict, optional): ``epoch``, ``best_loss`` and ``optimizer`` of a legacy
            ``*.pth`` checkpoint to resume the optimizer, the scheduler and the epoch counter from.
    """
    def __init__(self, net, args, legacy_resume: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.net = net
        self.args = args
        self.automatic_optimization = False
        self.legacy_resume = legacy_resume

        self.best_loss = 10e9 if legacy_resume is None else legacy_resume['best_loss']
        self.is_best = False
        self.criterion = None
        self.train_meters = None
        self.val_meters = None  # meters of the last validation, merged over all processes

        self._ddp_ready = False
        self._scaler = None
        self._use_amp = False
        self._nan_count = 0
        self._epochs_in_this_run = 0
        self._epoch_start_time = None
        self._train_time = None
        self._val_meters_local = None

    # ##################################################################################################################
    #  setup
    # ##################################################################################################################
    def setup(self, stage: Optional[str] = None) -> None:
        device = self.trainer.strategy.root_device
        if not self._ddp_ready:
            self.net.to(device)
            self.net.convert_memory_format(choice=self.args.mem_format)
            self.net.DDP(device.index if device.type == 'cuda' else None)
            self._ddp_ready = True
        self.criterion = set_criterion(reg_loss_weight=self.args.l1, device=device)

    def configure_optimizers(self):
        optimizer = init_optimizer(self.args, self.net)
        train_loader = self.trainer.datamodule.train_dataloader()
        scheduler = set_lr_scheduler(optimizer, BASE_BATCH_SIZE, train_loader, self.args)
        if self.legacy_resume is not None:
            self._restore_legacy_optimizer(optimizer, scheduler, len(train_loader))
        return [optimizer], [scheduler]

    def _restore_legacy_optimizer(self, optimizer, scheduler, steps_per_epoch):
        """Load the optimizer state of a legacy checkpoint and replay the scheduler up to its epoch."""
        print(f"=> loading optimizer from '{self.args.resume}'")
        optimizer.load_state_dict(self.legacy_resume['optimizer'])
        start_epoch = self.legacy_resume['epoch']
        for _ in range(start_epoch):
            if self.args.lr_type == 'step.epoch':
                scheduler.step()
            elif self.args.lr_type == 'reduce_on_plateau.epoch':
                scheduler.step(self.best_loss)
            if self.args.lr_type.endswith('step'):
                for _ in range(steps_per_epoch):
                    scheduler.step()

    def on_fit_start(self) -> None:
        if self.legacy_resume is not None and self.trainer.current_epoch == 0:
            # the epoch counter of a legacy checkpoint (a Lightning checkpoint restores it by itself)
            start_epoch = self.legacy_resume['epoch']
            progress = self.trainer.fit_loop.epoch_progress.current
            progress.ready = progress.started = progress.processed = progress.completed = start_epoch

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['best_loss'] = self.best_loss

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.best_loss = checkpoint.get('best_loss', self.best_loss)

    # ##################################################################################################################
    #  training
    # ##################################################################################################################
    def on_train_epoch_start(self) -> None:
        if self._epochs_in_this_run == MAX_EPOCHS_PER_RUN and not self.args.overfit:
            # the launcher restarts the stage from the last checkpoint (no "end of train" printed)
            sys.exit()
        self._epochs_in_this_run += 1
        self._epoch_start_time = datetime.now()
        epoch = self.current_epoch
        set_random(self.args.seed + epoch)
        recreate_train_iterator_if_created(self.trainer)

        scheduler = self.lr_schedulers()
        if self.args.lr_type == 'step.epoch':
            scheduler.step()
        elif self.args.lr_type == 'reduce_on_plateau.epoch':
            scheduler.step(self.best_loss)

        self.net.set_train_mode()
        # torch.cuda.amp works on GPUs only (on CPU the autocast/GradScaler are no-ops)
        amp = self.args.amp and self.device.type == 'cuda'
        self._nan_count, self._use_amp = 0, amp
        self._scaler = GradScaler() if amp else None
        self.train_meters = create_meters(TRAIN_METER_KEYS)
        self._train_time = time.time()
        torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def _total_loss(loss_dict):
        return loss_dict['rd_loss'] + loss_dict['reg_loss']

    def _forward_loss(self, x, beta, amp):
        loss_dict = self.net.train_forward_to_loss(x, self.criterion, beta, self.args, amp=amp)
        return loss_dict, self._total_loss(loss_dict)

    def _clip_gradients(self):
        for m in self.net.model_list:
            clip_grad_norm_(m.parameters(), GRAD_CLIP_NORM)

    def _any_process(self, flag: bool) -> bool:
        """True if ``flag`` is set on any process."""
        flags = self.all_gather(torch.tensor(float(flag), device=self.device))
        return bool(torch.any(flags > 0).item())

    def _register_nan_step(self):
        self._nan_count += 1
        if self._nan_count / self.trainer.num_training_batches > AMP_MAX_NAN_RATIO:
            self._use_amp = False

    def training_step(self, batch, batch_idx):
        _, x = batch
        args = self.args
        optimizer = self.optimizers()
        raw_optimizer = getattr(optimizer, 'optimizer', optimizer)
        scheduler = self.lr_schedulers()
        start_time = time.time()

        if args.lr_type.endswith('step'):
            scheduler.step()
        beta = self.net.beta_list[np.random.randint(0, len(self.net.beta_list))]
        loss_dict, loss = self._forward_loss(x, beta, amp=self._use_amp)

        # NaN loss on any process: redo the step in fp32
        self.trainer.strategy.barrier()
        overflow = has_nan([self.all_gather(loss.detach())])
        if overflow and self._use_amp:
            self._nan_count += 1
            loss_dict, loss = self._forward_loss(x, beta, amp=False)
            if self._nan_count / self.trainer.num_training_batches > AMP_MAX_NAN_RATIO:
                self._use_amp = False

        optimizer.zero_grad()
        if self._use_amp and not overflow:
            scaler = self._scaler
            self.manual_backward(scaler.scale(loss))
            scaler.unscale_(raw_optimizer)
            self._clip_gradients()
            found_inf = scaler._per_optimizer_states[id(raw_optimizer)]['found_inf_per_device']
            is_nan_grad = self._any_process(sum(v.item() for v in found_inf.values()) > 0)
            scale_factor = scaler.get_scale()
            if not is_nan_grad:
                optimizer.step()  # the same as scaler.step() for finite gradients
            else:
                self._register_nan_step()
            if scaler.get_scale() < AMP_MIN_SCALE:
                scaler.update(AMP_MIN_SCALE)
            else:
                scaler.update()
            if is_nan_grad:
                optimizer.zero_grad()
                loss_dict, loss = self._forward_loss(x, beta, amp=False)
                self.manual_backward(loss)
                self._clip_gradients()
                optimizer.step()
                scale_factor = 16.0
        else:
            self.manual_backward(loss)
            self._clip_gradients()
            optimizer.step()
            scale_factor = 32.0

        batch_size = x.shape[0]
        update_meters(
            batch_size, self.train_meters,
            dict({
                'Time': time.time() - start_time,
                'lr': raw_optimizer.param_groups[0]['lr'],
                'scale': scale_factor,
            }, **loss_dict))
        if self.global_rank == 0 and batch_idx % args.print_freq == 0:
            print_meters(self.train_meters,
                         keys=['Time', 'lr', 'rd_loss', 'reg_loss', 'scale'],
                         prefix=f'[{self.current_epoch:d}/{args.epochs:d}][{batch_idx:d}/{self.trainer.num_training_batches:d}] ')

    def _report_train_time(self) -> None:
        if self._train_time is not None and self.global_rank == 0:
            print(f'Train time of epoch {self.current_epoch} is {time.time() - self._train_time} seconds')
        self._train_time = None

    # ##################################################################################################################
    #  validation
    # ##################################################################################################################
    def on_validation_epoch_start(self) -> None:
        # the validation of an epoch starts right after its last training step
        self._report_train_time()
        self.net.set_eval_mode()
        self.net.zero_channel_wise_entropy()
        self._val_meters_local = create_meters(VAL_METER_KEYS)
        torch.cuda.empty_cache()
        gc.collect()

    def validation_step(self, batch, batch_idx):
        _, x = batch
        beta = self.net.beta_list[batch_idx % len(self.net.beta_list)]
        val_ret = self.net.val_forward_to_loss(x, self.criterion, beta, self.args, amp=False)
        if val_ret is None:
            return
        if len(self.args.rec_dir) > 0 and 'rec_data' in val_ret:
            self._store_reconstruction(val_ret['rec_data'])
        update_meters(x.shape[0], self._val_meters_local, val_ret)

    def _store_reconstruction(self, rec_data):
        import os

        from src.codec.common import Image
        img = Image.create_from_tensors(rec_data['xY'], rec_data['xCb'], rec_data['xCr'], [0, 255], color_space='yuv')
        os.makedirs(self.args.rec_dir, exist_ok=True)
        epoch = -1 if self.trainer.sanity_checking else self.current_epoch
        img.write_file(os.path.join(self.args.rec_dir, f"iter_{epoch:06d}.png"))

    def _merge_meters(self, meters):
        """Merge the averages of the meters over all processes (weighted by the counts)."""
        merged = create_meters(meters.keys())
        for key, meter in meters.items():
            local = torch.tensor([meter.avg, meter.count], dtype=torch.float32, device=self.device)
            gathered = self.all_gather(local).reshape(-1, 2)
            for avg, count in gathered.tolist():
                merged[key].update(avg, count)
        return merged

    def on_validation_epoch_end(self) -> None:
        self.val_meters = self._merge_meters(self._val_meters_local)
        if self.trainer.sanity_checking:
            # validation before training: reported only
            self.is_best = False
            return
        self.is_best = self.val_meters['Loss'].avg < self.best_loss
        if self.is_best:
            self.best_loss = self.val_meters['Loss'].avg
            if self.global_rank == 0:
                print(f'Best so far is epoch {self.current_epoch}')

    # ##################################################################################################################
    #  legacy checkpoints
    # ##################################################################################################################
    def save_legacy_checkpoint(self, optimizer, epoch: int, is_best: bool) -> None:
        """Write ``<epoch>.pth`` (and ``best.pth``) to ``--train_url`` in the format of the original training."""
        self.net.save_model_and_opt(self.args, epoch, optimizer, self.best_loss,
                                    model_prefix=str(epoch), is_best=is_best)
