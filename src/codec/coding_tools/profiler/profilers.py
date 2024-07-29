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

from .cpu_memory_usage import CpuMemoryUsage
from .gpu_memory_usage import GpuMemoryUsage
from .module_duration import ModuleDuration
from .profilers_interface import ProfilersInterface


class Profilers(ProfilersInterface):
    def _params_loaded(self):
        if self.module_duration:
            self.MODULE_DURATION = ModuleDuration()
        if self.gpu_memory_usage:
            self.GPU_MEMORY_USAGE = GpuMemoryUsage()
        if self.cpu_memory_usage:
            self.CPU_MEMORY_USAGE = CpuMemoryUsage()
        self.LEVEL = 0
        self.ACTIVE_STOPWATCHES = set()

    def set_directory(self, path):
        '''Set directory for all the profilers to save their data'''
        if self.module_duration:
            self.MODULE_DURATION.set_directory(path)
        if self.gpu_memory_usage:
            self.GPU_MEMORY_USAGE.set_directory(path)
        if self.cpu_memory_usage:
            self.CPU_MEMORY_USAGE.set_directory(path)

    def add_event(self, event):
        '''Start event that has no duration for all the profilers'''
        if self.module_duration:
            self.MODULE_DURATION.add_event(self.LEVEL * '    ' + event)
        if self.gpu_memory_usage:
            self.GPU_MEMORY_USAGE.add_event(self.LEVEL * '    ' + event)
        if self.cpu_memory_usage:
            self.CPU_MEMORY_USAGE.add_event(self.LEVEL * '    ' + event)

    def start(self, event, stopwatch=False):
        '''Start event that has duration for all the profilers'''
        #print(f"Start {event} at level {self.LEVEL}")
        if stopwatch:
            self.ACTIVE_STOPWATCHES.add(event)
        if self.module_duration:
            self.MODULE_DURATION.start(self.LEVEL * '    ' + event, stopwatch)
        if not stopwatch and self.gpu_memory_usage:
            self.GPU_MEMORY_USAGE.start(self.LEVEL * '    ' + event)
        if not stopwatch and self.cpu_memory_usage:
            self.CPU_MEMORY_USAGE.start(self.LEVEL * '    ' + event)
        if event not in self.ACTIVE_STOPWATCHES:
            self.LEVEL += 1

    def finish(self, event, stopwatch=False):
        '''Finish event that has duration for all the profilers'''
        if not stopwatch and event not in self.ACTIVE_STOPWATCHES:
            self.LEVEL -= 1
        if self.module_duration:
            self.MODULE_DURATION.finish(self.LEVEL * '    ' + event, stopwatch)
        if not stopwatch and self.gpu_memory_usage and event not in self.ACTIVE_STOPWATCHES:
            self.GPU_MEMORY_USAGE.finish(self.LEVEL * '    ' + event)
        if not stopwatch and self.cpu_memory_usage and event not in self.ACTIVE_STOPWATCHES:
            self.CPU_MEMORY_USAGE.finish(self.LEVEL * '    ' + event)
        if not stopwatch and event in self.ACTIVE_STOPWATCHES:
            self.ACTIVE_STOPWATCHES.discard(event)
        #print(f"Finished {event} at level {self.LEVEL}")5

    def save_results(self, filename=None):
        '''Save results for all the profilers'''
        if filename is None:
            if self.module_duration:
                self.MODULE_DURATION.save_results()
            if self.gpu_memory_usage:
                self.GPU_MEMORY_USAGE.save_results()
            if self.cpu_memory_usage:
                self.CPU_MEMORY_USAGE.save_results()
        else:
            if self.module_duration:
                self.MODULE_DURATION.save_results(filename)
            if self.gpu_memory_usage:
                self.GPU_MEMORY_USAGE.save_results(filename)
            if self.cpu_memory_usage:
                self.CPU_MEMORY_USAGE.save_results(filename)
