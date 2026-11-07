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
"""Compatibility layer between Lightning versions.

The repository pins ``torch==1.10.2``; the newest Lightning release supporting it is
``pytorch-lightning==1.9.5``. The code also works with Lightning 2.x (``lightning`` or
``pytorch-lightning`` packages) when torch is upgraded.
"""

import inspect

try:  # Lightning >= 2.0, unified package
    import lightning.pytorch as pl
except ImportError:  # pytorch-lightning package (1.9.x or 2.x)
    import pytorch_lightning as pl

__all__ = ['pl', 'PL_MAJOR_VERSION', 'no_sampler_replacement_kwargs', 'recreate_train_iterator_if_created']

PL_MAJOR_VERSION = int(pl.__version__.split('.')[0])


def no_sampler_replacement_kwargs() -> dict:
    """Trainer kwargs disabling the automatic ``DistributedSampler`` injection.

    The data module creates the distributed samplers itself (as the original training did).
    The argument is ``replace_sampler_ddp`` in Lightning 1.x and ``use_distributed_sampler`` in 2.x.
    """
    params = inspect.signature(pl.Trainer.__init__).parameters
    if 'use_distributed_sampler' in params:
        return {'use_distributed_sampler': False}
    return {'replace_sampler_ddp': False}


def recreate_train_iterator_if_created(trainer) -> None:
    """Re-create the iterator of the training dataloader if it was created before ``on_train_epoch_start``.

    Lightning 1.x creates the iterator of an epoch after ``on_train_epoch_start``. Lightning 2.x does
    the same, except for epoch 0 and for an epoch restarted from a checkpoint: their iterator is
    created before the hook. Creating an iterator draws the seed of the dataloader from the global
    random generator, so such an iterator is re-created after the random generators are re-seeded in
    ``on_train_epoch_start`` to get the same data with both versions. Call it from that hook.
    """
    if PL_MAJOR_VERSION < 2:
        return
    fit_loop = trainer.fit_loop
    if not (trainer.current_epoch == 0 or fit_loop.epoch_loop.restarting):
        return  # the epoch loop creates the iterator after the hook
    fetcher = getattr(fit_loop, '_data_fetcher', None)
    if fetcher is not None and getattr(fetcher, 'iterator', None) is not None:
        iter(fetcher)
