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

import os
from datetime import datetime

import pandas as pd


class ModuleDuration:
    def __init__(self):
        self.table = pd.DataFrame(
            {
                '_START_': datetime.now(),
                'Event': 'Initialization',
                'Start time': datetime.now().time(),
                'Duration': '',
                'Duration, sec': ''
            },
            index=[0])
        self.directory = ''
        self.buffer = dict()

    def set_directory(self, path):
        '''Set directory for saving collector's data'''
        os.makedirs(os.path.join(path, 'MODULES_DURATION'), exist_ok=True)
        self.directory = os.path.join(path, 'MODULES_DURATION')

    def add_event(self, event):
        '''Start event that has no duration'''
        self.buffer[event] = {
            '_START_': datetime.now(),
            'Event': event,
            'Duration': '',
            'Duration, sec': ''
        }
        self.buffer[event]['Start time'] = self.buffer[event]['_START_']
        self.table = self.table.append(self.buffer[event], ignore_index=True)

    def start(self, event, stopwatch=False):
        '''Start event that has duration'''
        if not stopwatch or event not in self.buffer:
            self.buffer[event] = {'_START_': datetime.now(), 'Event': event}
            self.buffer[event]['Start time'] = self.buffer[event]['_START_']
            # Avoid format conversion for zero
            self.buffer[event][
                'Duration'] = self.buffer[event]['Start time'] - self.buffer[event]['Start time']
        else:
            self.buffer[event]['Start time'] = datetime.now()

    def finish(self, event, stopwatch=False):
        '''Finish event that has duration'''
        if stopwatch or 'Finish time' not in self.buffer[event]:
            self.buffer[event]['Finish time'] = datetime.now()
            self.buffer[event][
                'Duration'] += self.buffer[event]['Finish time'] - self.buffer[event]['Start time']
        if not stopwatch:
            self.buffer[event]['Duration, sec'] = round(
                self.buffer[event]['Duration'].seconds +
                self.buffer[event]['Duration'].microseconds * 10**-6, 2)
            self.buffer[event]['Start time'] = self.buffer[event]['_START_'].time()
            del self.buffer[event]['Finish time']
            self.table = self.table.append(self.buffer[event], ignore_index=True)
            del self.buffer[event]

    def save_results(self, filename='MODULES_DURATION'):
        '''Save results to the directory with name output_name filename.CSV'''
        output_name = filename + '.CSV'
        if self.directory:
            self.table.sort_values(by='_START_').drop(columns=['_START_']).to_csv(os.path.join(
                self.directory, output_name),
                                                                                  index=False)
        else:
            self.table.sort_values(by='_START_').drop(columns=['_START_']).to_csv(output_name,
                                                                                  index=False)
