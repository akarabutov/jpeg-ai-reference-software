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


class Dataset:
    def __init__(self):
        self.instances = []
        self._idx = None
        self.kwargs = None

    def getInstanceID(self):
        """
        returns a unique identifier for the current instance
        e.g. return filename
        """
        raise NotImplementedError()

    def getInstance(self, idx):
        """
        returns a unique identifier for the current instance
        e.g. return filename
        """
        raise NotImplementedError()

    def load(self, idx):
        """
        loads data instance corresponding to idx
        """
        raise NotImplementedError()

    def _checkProps(self, inst: dict, props: dict):
        """
        check for each data inst if properties match provided requirements
        """
        for key, value in props.items():
            if key not in inst:
                return False

            if type(inst[key]) == dict:
                if not self._checkProps(inst[key], value):
                    return False
                continue

            if type(value) != list:
                value = [value]

            if len(value) > 0 and not inst[key] in value:
                return False

        return True

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self

    def __iter__(self, **kwargs):
        if kwargs:
            self.kwargs = kwargs
        self._idx = None
        return self

    def __next__(self):
        if self._idx is None:
            self._idx = -1

        if self._idx < len(self.instances) - 1:
            self._idx += 1
            if self.kwargs:
                return self.load(self._idx, **self.kwargs)
            else:
                return self.load(self._idx)
        else:
            raise StopIteration

    def __len__(self):
        return len(self.instances)
