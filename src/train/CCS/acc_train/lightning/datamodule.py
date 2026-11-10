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
"""Training and validation data of the CCS training."""

from typing import Optional

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler

from src.train.CCS.acc_train.dataset import (CodecDataset, CustomCrop, CustomResize,
                                             CustomToTensor)

from .compat import pl

__all__ = ['CCSDataModule', 'StackCrops', 'train_transform']


class StackCrops:
    """Convert a list of PIL crops to a ``[num_crops, C, H, W]`` tensor (values in [0, 255])."""
    def __init__(self):
        self._to_tensor = CustomToTensor()

    def __call__(self, crops):
        return torch.stack([self._to_tensor(crop) for crop in crops])

    def __repr__(self):
        return self.__class__.__name__ + '()'


def train_transform(args):
    if args.overfit:
        return CustomToTensor()
    return transforms.Compose([
        CustomResize([1024, 512]),
        transforms.RandomHorizontalFlip(p=0.5),
        CustomCrop(args.crop_size, args.crop_number),
        StackCrops(),
    ])


class _WorkersReleasingIterator:
    """Iterator of a dataloader which stops the worker processes right after the last batch is taken."""
    def __init__(self, iterator, length: int):
        self._iterator = iterator
        self._length = length
        self._remaining = length

    def __iter__(self):
        return self

    def __len__(self):
        return self._length

    def __next__(self):
        if self._iterator is None:
            raise StopIteration
        batch = next(self._iterator)
        self._remaining -= 1
        if self._remaining <= 0:
            self._iterator = None  # the iterator is deleted: its worker processes are shut down
        return batch

    def __getattr__(self, name):
        # called only for attributes which are not found here
        if name == '_iterator' or self._iterator is None:
            raise AttributeError(name)
        return getattr(self._iterator, name)


class WorkersReleasingDataLoader(DataLoader):
    """``DataLoader`` which releases its worker processes as soon as an epoch of data is consumed.

    The worker processes of a ``DataLoader`` exit when its iterator is exhausted or deleted. Lightning knows the
    number of batches and does not ask for a batch after the last one, so the workers of the training loader
    stayed alive during the validation and the workers of the validation loader during a part of the next
    epoch. A worker is forked from the training process; while it is alive every page of the weights and of the
    optimizer state changed by the training is duplicated (copy-on-write), i.e. on CPU the lingering workers
    cost a copy of the model state each. Here the iterator is deleted right after the last batch.
    """
    def __iter__(self):
        iterator = super().__iter__()
        if self.num_workers == 0:
            return iterator
        return _WorkersReleasingIterator(iterator, len(self))


class CCSDataModule(pl.LightningDataModule):
    """Data of one training stage.

    The samplers are created here (``DistributedSampler`` with the default seed, shuffling for the
    training set only), so the Trainer must be created with the automatic sampler replacement
    disabled (see ``compat.no_sampler_replacement_kwargs``). The validation set is processed with
    batch size 1 and its size has to be divisible by the number of processes, so that every image
    is validated exactly once.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_dataset: Optional[CodecDataset] = None
        self.val_dataset: Optional[CodecDataset] = None

    def _world(self):
        if self.trainer is not None:
            return self.trainer.world_size, self.trainer.global_rank
        return 1, 0

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is None:
            self.train_dataset = CodecDataset(self.args.data_dir, self.args.lst, train_transform(self.args))
        if self.val_dataset is None and self.args.val_lst:
            self.val_dataset = CodecDataset(self.args.val_data_dir, self.args.val_lst, CustomToTensor())
            world_size, _ = self._world()
            assert len(self.val_dataset) % world_size == 0, \
                f'Size of the validation set ({len(self.val_dataset)}) must be divisible by the number of processes ({world_size})'

    def train_dataloader(self) -> DataLoader:
        world_size, rank = self._world()
        sampler = DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        return WorkersReleasingDataLoader(self.train_dataset,
                                          batch_size=self.args.batch_size,
                                          num_workers=self.args.workers,
                                          pin_memory=True,
                                          sampler=sampler)

    def val_dataloader(self) -> DataLoader:
        world_size, rank = self._world()
        sampler = DistributedSampler(self.val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        # One worker reads the next picture while the current one is processed. With `--workers 0` no worker
        # process is used at all: a worker is forked from the training process, and while it is alive every
        # page of the weights changed by the training is duplicated (copy-on-write), which matters when the
        # model is trained on CPU with little memory.
        return WorkersReleasingDataLoader(self.val_dataset,
                                          batch_size=1,
                                          num_workers=min(1, self.args.workers),
                                          pin_memory=False,
                                          sampler=sampler)

    def steps_per_epoch(self) -> int:
        """Number of training batches per process and epoch."""
        return len(self.train_dataloader())
