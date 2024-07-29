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
import psutil


class CpuMemoryUsage:
    def __init__(self):
        memory = psutil.virtual_memory()
        total = memory.total / 1024 / 1024
        used = memory.used / 1024 / 1024
        self.table = pd.DataFrame(
            {
                '_START_': datetime.now(),
                'Event': 'Initialization',
                'Total, MB': total,
                'Start CPU memory usage, MB': used,
                'Finish CPU memory usage, MB': '',
                'Difference, MB': ''
            },
            index=[0])
        self.directory = ''
        self.buffer = dict()

    def set_directory(self, path):
        '''Set directory for saving collector's data'''
        os.makedirs(os.path.join(path, 'CPU_MEMORY_USAGE'), exist_ok=True)
        self.directory = os.path.join(path, 'CPU_MEMORY_USAGE')

    def add_event(self, event):
        '''Start event that has no duration'''
        memory = psutil.virtual_memory()
        total = memory.total / 1024 / 1024
        used = memory.used / 1024 / 1024
        self.buffer[event] = {
            '_START_': datetime.now(),
            'Event': event,
            'Total, MB': total,
            'Start CPU memory usage, MB': used,
            'Finish CPU memory usage, MB': '',
            'Difference, MB': ''
        }
        self.table = self.table.append(self.buffer[event], ignore_index=True)

    def start(self, event):
        '''Start event that has duration'''
        memory = psutil.virtual_memory()
        total = memory.total / 1024 / 1024
        used = memory.used / 1024 / 1024
        self.buffer[event] = {
            '_START_': datetime.now(),
            'Event': event,
            'Total, MB': total,
            'Start CPU memory usage, MB': used
        }

    def finish(self, event):
        '''Finish event that has duration'''
        memory = psutil.virtual_memory()
        used = memory.used / 1024 / 1024
        self.buffer[event]['Finish CPU memory usage, MB'] = used
        self.buffer[event]['Difference, MB'] = self.buffer[event][
            'Finish CPU memory usage, MB'] - self.buffer[event]['Start CPU memory usage, MB']
        self.table = self.table.append(self.buffer[event], ignore_index=True)

    def save_results(self, filename='CPU_MEMORY_USAGE'):
        output_name = filename + '.CSV'
        '''Save results to the directory with name output_name filename.CSV'''
        if self.directory:
            self.table.sort_values(by='_START_').drop(columns=['_START_']).to_csv(os.path.join(
                self.directory, output_name),
                                                                                  index=False)
        else:
            self.table.sort_values(by='_START_').drop(columns=['_START_']).to_csv(output_name,
                                                                                  index=False)
