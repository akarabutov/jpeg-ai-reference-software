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

"""Training of one stage of the CCS codec models with PyTorch Lightning.

The script is started by ``scripts/acc_train_scripts/acc_train_local.py`` with ``torch.distributed.run``
(one process per GPU); it can also be started directly as one process. The last lines of its
stdout must contain ``end of train``: the launcher restarts a stage which did not print it.
"""

import os
from datetime import timedelta
from pathlib import Path

import torch

from ..lightning import (AutomaticTesting, CCSDataModule, CCSLightningModule, DirectCheckpointIO,
                         PerSubmoduleDDPStrategy, StageOutputs, pl)
from ..lightning.callbacks import LAST_CKPT_NAME
from ..lightning.compat import no_sampler_replacement_kwargs
from ..model import Net
from ..utils import get_args, set_random

PROCESS_GROUP_TIMEOUT = timedelta(seconds=7200)


def _devices_and_nodes():
    """Processes per node and number of nodes, as set by ``torch.distributed.run`` (1 and 1 otherwise)."""
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', world_size))
    return local_world_size, max(1, world_size // local_world_size)


def _load_legacy_resume(args):
    """Optimizer state, epoch and best loss of ``--resume`` when ``--resume_opt`` is set."""
    if not args.resume_opt:
        return None
    if not os.path.isfile(args.resume):
        raise ValueError(f'{args.resume} is not file')
    checkpoint = torch.load(args.resume, map_location='cpu')
    return {key: checkpoint[key] for key in ('epoch', 'best_loss', 'optimizer')}


def _select_lightning_checkpoint(args, legacy_resume):
    """``last.ckpt`` of the stage if the training is resumed and it is not older than ``--resume``."""
    ckpt_path = os.path.join(args.train_url, LAST_CKPT_NAME)
    if legacy_resume is None or not os.path.isfile(ckpt_path):
        return None
    ckpt_epoch = torch.load(ckpt_path, map_location='cpu').get('epoch', -1)
    # Lightning stores the finished epoch, a legacy checkpoint the next one
    if ckpt_epoch + 1 < legacy_resume['epoch']:
        return None
    return ckpt_path


def _create_trainer(args, callbacks, logger, num_sanity_val_steps=0):
    devices, num_nodes = _devices_and_nodes()
    return pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=devices,
        num_nodes=num_nodes,
        strategy=PerSubmoduleDDPStrategy(timeout=PROCESS_GROUP_TIMEOUT),
        precision=32,  # mixed precision is handled by CCSLightningModule (custom AMP fallback)
        max_epochs=args.epochs,
        logger=logger,
        callbacks=callbacks,
        plugins=[DirectCheckpointIO()],  # last.ckpt is written without a copy in memory
        num_sanity_val_steps=num_sanity_val_steps,
        enable_checkpointing=False,  # StageOutputs writes the checkpoints
        enable_progress_bar=False,
        enable_model_summary=False,
        inference_mode=False,
        default_root_dir=args.train_url,
        **no_sampler_replacement_kwargs(),
    )


def _create_logger(args):
    stage = Path(args.train_url).parent.name
    tensorboard_root = Path(args.train_url).parents[1] / 'tensorboard'
    return pl.loggers.TensorBoardLogger(save_dir=str(tensorboard_root),
                                        name=stage,
                                        version=str(args.beta),
                                        default_hp_metric=False)


def _create_net(args):
    net = Net(args)
    if args.resume:
        net.resume_weight(args.resume, False)
    return net


def run_training(args, legacy_resume):
    ckpt_path = _select_lightning_checkpoint(args, legacy_resume)
    if ckpt_path is not None:
        print(f'=> resuming the training state from {ckpt_path}')
        legacy_resume = None

    datamodule = CCSDataModule(args)
    module = CCSLightningModule(_create_net(args), args, legacy_resume=legacy_resume)
    callbacks = [StageOutputs(args.train_url)]
    if args.use_automatic_testing:
        callbacks.append(AutomaticTesting(args, args.automatic_testing_epoch_period))
    # validation before training, unless the training of this stage is resumed
    num_sanity_val_steps = 0 if args.resume_opt else -1
    trainer = _create_trainer(args, callbacks, _create_logger(args), num_sanity_val_steps)
    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)


def main():
    print('start of train')
    args = get_args()
    os.makedirs(args.train_url, exist_ok=True)
    # the same initialisation of the weights in every process
    set_random(args.seed)
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    run_training(args, _load_legacy_resume(args))
    print('end of train')


if __name__ == '__main__':
    main()
