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
"""Callbacks writing the outputs of a training stage."""

import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import torch

from src.train.CCS.acc_train.utils.train_utils import ResultPrinter

from .compat import pl

__all__ = ['StageOutputs', 'AutomaticTesting', 'LAST_CKPT_NAME', 'KEEP_FILES']

LAST_CKPT_NAME = 'last.ckpt'
KEEP_FILES = ('best.pth', 'val_results.json', LAST_CKPT_NAME)


class StageOutputs(pl.Callback):
    """Outputs of a stage in ``--train_url``, compatible with the original training scripts.

    * validation table (printed and stored to ``val_results.json``) with the rows
      ``Before train``, ``Epoch:[e]`` and ``best``;
    * tensorboard scalars ``<meter>/train`` and ``<meter>/validation`` per epoch;
    * ``<epoch>.pth`` and ``best.pth`` legacy checkpoints (the older files are removed, as the launcher
      only needs the last epoch for a restart and the best one for the next stage);
    * ``last.ckpt`` - full Lightning checkpoint (optimizer, scheduler, loop state) used to resume
      after a crash.
    """
    def __init__(self, train_url: str):
        super().__init__()
        self.train_url = train_url
        self.printer = None
        self.time_all_epochs = timedelta()

    # validation table
    def _rank_zero_store_table(self, trainer):
        if trainer.is_global_zero:
            self.printer.print()
            output_fn = os.path.join(self.train_url, 'val_results.json')
            self.printer.store(output_fn)
            return output_fn
        return None

    def on_fit_start(self, trainer, pl_module):
        os.makedirs(self.train_url, exist_ok=True)
        if trainer.is_global_zero and trainer.logger is not None:
            layout = {
                'Train vs. validation loss': {
                    'loss': ['Multiline', ['rd_loss/train', 'reg_loss/train', 'Loss/validation']],
                },
            }
            trainer.logger.experiment.add_custom_scalars(layout)

    def on_validation_end(self, trainer, pl_module):
        meters = pl_module.val_meters
        if self.printer is None:
            self.printer = ResultPrinter(epoch_str='Keys', headers=list(meters.keys()))
        if trainer.sanity_checking:
            self.printer.insert_row(meters, 'Before train', 'avg')
            output_fn = self._rank_zero_store_table(trainer)
            if output_fn is not None:
                print(f'Stored validation results to a file {output_fn}')
            return
        self.printer.insert_row(meters, f'Epoch:[{trainer.current_epoch}]', 'avg')
        if pl_module.is_best:
            self.printer.insert_row(meters, 'best', 'avg')

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        trainer.strategy.barrier()
        if trainer.is_global_zero:
            self._rank_zero_store_table(trainer)
            self._update_tensorboard(trainer, pl_module, epoch)
            self._cleanup()
            pl_module.save_legacy_checkpoint(trainer.optimizers[0], epoch, pl_module.is_best)
        # every process has to call it (rank 0 writes the file)
        trainer.save_checkpoint(os.path.join(self.train_url, LAST_CKPT_NAME))
        if trainer.is_global_zero:
            time_for_this_epoch = datetime.now() - pl_module._epoch_start_time
            print(f'Epoch {epoch} took {time_for_this_epoch}')
            self.time_all_epochs += time_for_this_epoch

    def on_fit_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            print(f'Total time train: {self.time_all_epochs}')

    @staticmethod
    def _update_tensorboard(trainer, pl_module, epoch):
        if trainer.logger is None:
            return
        writer = trainer.logger.experiment
        for key, meter in pl_module.train_meters.items():
            if key != 'Time':
                writer.add_scalar(f'{key}/train', meter.avg, epoch)
        for key, meter in pl_module.val_meters.items():
            if key != 'Time':
                writer.add_scalar(f'{key}/validation', meter.avg, epoch)
        writer.flush()

    def _cleanup(self):
        # the launcher copies what it needs; keep only the best model, the results and the
        # Lightning checkpoint (it is overwritten right after)
        for path in os.listdir(self.train_url):
            abs_path = os.path.join(self.train_url, path)
            if os.path.isdir(abs_path):
                shutil.rmtree(abs_path)
            elif path not in KEEP_FILES:
                os.remove(abs_path)


class AutomaticTesting(pl.Callback):
    """Encode/decode the test set with ``best.pth`` every ``period`` epochs and after the last epoch.

    The results are stored to ``<train_url>/<epoch>_test/<beta>``. Only the first process runs the
    test, the others wait for it.
    """
    def __init__(self, args, period: int):
        super().__init__()
        self.args = args
        self.period = period
        self.time_all_tests = timedelta()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if trainer.is_global_zero:
            start_time_test = datetime.now()
            if epoch % self.period == 0 or self.args.epochs - 1 == epoch:
                self._run_test(epoch)
            time_for_this_test = datetime.now() - start_time_test
            print(f'Test {epoch} took {time_for_this_test}')
            self.time_all_tests += time_for_this_test
        trainer.strategy.barrier()

    def on_fit_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            print(f'Total time test: {self.time_all_tests}')

    def _run_test(self, epoch):
        from scripts.acc_train_scripts.utils import copy_model_for_test, run_test
        from scripts.split_cp import split_cp

        args = self.args
        proj_dir = Path(__file__).parents[5]
        with tempfile.TemporaryDirectory() as models_dir_path:
            print(f'Doing inference test for epoch {epoch}')
            cfgs = [os.path.join(proj_dir, 'cfg', x) for x in args.cfg_path]
            model_path_tmp = os.path.join(models_dir_path, 'VM_tmp')
            copy_model_for_test(model_path_tmp, args.beta, epoch, args.train_url, create_missing_models=True)
            op_list = list(set(args.vae_encoder_type_list + args.vae_decoder_type_list))
            for fn in os.listdir(model_path_tmp):
                data = torch.load(os.path.join(model_path_tmp, fn))
                split_cp(data, fn, op_list, models_dir_path)
            test_results_dir = os.path.join(args.train_url, f'{epoch}_test', f'{args.beta}')
            run_test(test_results_dir, epoch, args.test_data_dir, models_dir_path, cfgs, os.environ.copy(), args, args.beta)
