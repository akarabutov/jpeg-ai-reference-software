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
"""Writing of Lightning checkpoints with a low memory footprint."""

import os
from typing import Any, Dict, Optional

import torch

from .compat import pl

__all__ = ['DirectCheckpointIO']


class DirectCheckpointIO(pl.plugins.io.TorchCheckpointIO):
    """Checkpoint IO which serializes a checkpoint directly to the (local) file.

    The default IO of Lightning serializes the checkpoint to a buffer in memory and copies the buffer once more
    before writing it. A checkpoint of the CCS model with the optimizer state is about 640 MB, so every
    ``last.ckpt`` cost about 1.3 GB of extra memory. Here the checkpoint is written to a temporary file, which then
    replaces the target (so a crash while writing does not leave a broken checkpoint).
    """
    def save_checkpoint(self, checkpoint: Dict[str, Any], path, storage_options: Optional[Any] = None) -> None:
        path = str(path)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        tmp_path = path + '.part'
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
