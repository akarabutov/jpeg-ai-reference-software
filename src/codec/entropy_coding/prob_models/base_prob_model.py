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

import numpy as np


class BaseProbModel:
    """BaseProbModel
    """
    tiny = np.float32(1e-10)
    sqrt = 2**0.5

    def __init__(self, name, symbol_num, *args):
        self.name = name
        self.params = dict()
        self._init_symbol_range(symbol_num)

    # ##################################################################################################################
    #  __init__ methods
    # ##################################################################################################################
    def _init_model_params(self, *args):
        raise NotImplementedError

    def _init_symbol_range(self, symbol_num):
        self.symbol_min = 0
        self.symbol_max = symbol_num - 1
        self.symbol_num = symbol_num

    # ##################################################################################################################
    #  main methods
    # ##################################################################################################################
    def get_model_params(self, key=None):
        if key is None:
            return self.params

        if key in self.params:
            return self.params[key]
        else:
            raise AssertionError('Invalid param with key={}'.format(key))

    def to_ctypes(self, *args):
        """

        Args:
            *args:

        Returns:
            cptr, size, ... :
        """
        raise NotImplementedError

    # ##################################################################################################################
    #  service methods
    # ##################################################################################################################
    def __check_symbol(self, symbol):
        if ((symbol >= self.symbol_min) & (symbol <= self.symbol_max)).all():
            return
        else:
            raise AssertionError('Symbol out of range=[{}, {}]'.format(
                self.symbol_min, self.symbol_max))

    def __str__(self):
        """Returns the name of distribution

        Returns:
            prob_model_type
        """
        return self.name


class ProbModelTypes:
    types = {
        0: 'CustomProbModel',
        1: 'AgmmProbModel',
        2: 'GolombProbModel',
        3: 'HistProbModel',
        4: 'AsgmProbModel',
        5: 'BypassProbModel',
    }

    @classmethod
    def index2name(cls, index: int):
        if index not in cls.types.keys():
            raise KeyError('Unsupported probability model with index={}'.format(index))
        return cls.types[index]

    @classmethod
    def set_prob_model(cls, name: str):
        if name not in cls.types.values():
            raise KeyError('Unsupported probability model with index={}'.format(type))
        return name
