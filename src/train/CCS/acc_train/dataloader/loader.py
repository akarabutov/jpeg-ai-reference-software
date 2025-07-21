# The copyright in this software is being made available under the BSD
# License, included below. This software may be subject to other third party
# and contributor rights, including patent rights, and no such rights are
# granted under this license.
#
# Copyright (c) 2010-2022, ITU/ISO/IEC
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

import torch
import torch.distributed as dist
import torchvision.transforms as transforms

from src.train.CCS.acc_train.dataset import (CodecDataset, CustomCrop, CustomResize,
                                             CustomToTensor)

__all__ = ['get_data_loader']


def get_data_loader(args):
    tansform = CustomToTensor() if args.overfit else transforms.Compose([
            CustomResize([1024, 512]),
            transforms.RandomHorizontalFlip(p=0.5),
            CustomCrop(args.crop_size, args.crop_number),
            transforms.Lambda(
                lambda crops: torch.stack([CustomToTensor()(crop) for crop in crops]))
        ])
    train_dataset = CodecDataset(
        args.data_dir, args.lst, tansform
        )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler)

    val_dataset = CodecDataset(args.val_data_dir, args.val_lst, CustomToTensor())
    assert len(val_dataset) % dist.get_world_size() == 0
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=1,
                                             num_workers=1,
                                             pin_memory=False,
                                             sampler=val_sampler)
    return train_loader, val_loader
