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

import logging


class Logger:
    levels = ['debug', 'info', 'warn', 'error', 'critical']
    mapper = {
        'debug': {
            'index': 0,
            'logging': logging.DEBUG
        },
        'info': {
            'index': 1,
            'logging': logging.INFO
        },
        'warn': {
            'index': 2,
            'logging': logging.WARNING
        },
        'error': {
            'index': 3,
            'logging': logging.ERROR
        },
        'critical': {
            'index': 4,
            'logging': logging.CRITICAL
        },
    }

    def __init__(self, name, level: str):
        self.name = name
        self._level = level

    # ##################################################################################################################
    #  @property methods
    # ##################################################################################################################
    #  level
    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, level: str):
        self._level = level

    # ##################################################################################################################
    #  Public methods
    # ##################################################################################################################
    def debug(self, msg):
        self._print('debug', msg)

    def info(self, msg):
        self._print('info', msg)

    def warn(self, msg):
        self._print('warn', msg)

    def error(self, msg):
        self._print('error', msg)

    def critical(self, msg):
        self._print('critical', msg)

    def _print(self, this_level: str, msg):
        this_index = self.mapper[this_level]['index']
        self_index = self.mapper[self.level]['index']
        if this_index < self_index:
            return

        string = 'Logger[{}] with Level={}: {}'.format(self.name, this_level, msg)
        print(string)


if __name__ == '__main__':
    debugLogger = Logger(name='debug', level=0)

    debugLogger.debug('debug')
    debugLogger.info('info')
    debugLogger.warn('warn')
    debugLogger.error('error')
    debugLogger.critical('critical')

    import torch

    channels = 10
    num_symbols = 100
    input_chs = 2 * channels

    ans = torch.arange(10, device='cpu')  # shape=[channels, ], value=[0, channels-1]
    print('ans[0]: {}\n{}\n'.format(ans.shape, ans))

    ans = ans[...,
              None] * num_symbols  # shape=[channels, 1], value=[0, (channels-1) * num_symbols]
    print('ans[1]: {}\n{}\n'.format(ans.shape, ans))

    ans = ans.repeat(1, input_chs // channels).flatten()
    print('ans[2]: {}\n{}\n'.format(ans.shape, ans))
