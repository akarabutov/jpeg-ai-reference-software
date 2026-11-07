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
        return DataLoader(self.train_dataset,
                          batch_size=self.args.batch_size,
                          num_workers=self.args.workers,
                          pin_memory=True,
                          sampler=sampler)

    def val_dataloader(self) -> DataLoader:
        world_size, rank = self._world()
        sampler = DistributedSampler(self.val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        return DataLoader(self.val_dataset,
                          batch_size=1,
                          num_workers=1,
                          pin_memory=False,
                          sampler=sampler)

    def steps_per_epoch(self) -> int:
        """Number of training batches per process and epoch."""
        return len(self.train_dataloader())
