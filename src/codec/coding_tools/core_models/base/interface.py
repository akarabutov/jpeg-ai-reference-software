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

from typing import Tuple

from ....common import Decisions, Image
from ...interfaces import ToolEngine
##
from .params import CoreModelParams


class CoreModelBase(ToolEngine):
    def __init__(self, *args, **kwargs):
        """Init core model
            * alignment (int)  : number of pixels on which the input image should be aligned
        """
        super(CoreModelBase, self).__init__(*args, **kwargs)
        self.__params_core_model = CoreModelParams()
        self._alignment = kwargs.get('alignment', 1)

    def get_internal_data_range(self) -> Tuple:
        raise NotImplementedError

    def compress(self, img: Image, latent_space: Decisions = None) -> Decisions:
        """
        Placeholder with delegation argument list:
        img is a original img, which have to be compressed
        latent_space is a dict with already done latent_space. If it presented it shouldn't be recomputed
        """
        raise NotImplementedError

    def postprocess_decisions(self, decisions: Decisions, h: int, w: int) -> Decisions:
        return decisions

    def get_alignment_size(self):
        return self._alignment
