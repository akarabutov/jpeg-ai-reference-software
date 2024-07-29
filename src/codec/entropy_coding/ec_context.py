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

from contextlib import ContextDecorator


class ECSettingContext(ContextDecorator):
    def __init__(self, module: "ECModule", stream_part: str = None, stream_base_comp: int = None, region_idx: int = 0):
        self.stream_part = stream_part
        self.stream_part_old = None
        self.region_idx = region_idx
        self.region_idx_old = None
        self.stream_base_comp = stream_base_comp
        self.stream_base_comp_old = None
        self.module = module

    def __enter__(self):
        self.stream_part_old = getattr(self.module, "stream_part", '')
        self.stream_base_comp_old = getattr(self.module, "stream_base_comp", 0)
        self.region_idx_old = getattr(self.module, "region_idx", 0)
        if self.stream_part is not None:
            setattr(self.module, "stream_part", self.stream_part)
        if self.stream_base_comp is not None:
            setattr(self.module, "stream_base_comp", self.stream_base_comp)
        if self.region_idx is not None:
            setattr(self.module, "region_idx", self.region_idx)
        return self

    def __exit__(self, *exc):
        setattr(self.module, "region_idx", self.region_idx_old)
        setattr(self.module, "stream_part", self.stream_part_old)
        setattr(self.module, "stream_base_comp", self.stream_base_comp_old)
        return False
