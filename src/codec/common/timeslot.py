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

from datetime import datetime


class Timeslot(object):
    """ Timeslot for metering time.
    """
    def __init__(self):
        self.bgn_time = None
        self.end_time = None
        self.gap_time = None

    # setter
    def set_bgn_time(self):
        self.bgn_time = datetime.now()

    def set_end_time(self):
        self.end_time = datetime.now()
        self.gap_time = self.end_time - self.bgn_time

    def to_seconds(self):
        total_seconds = self.gap_time.total_seconds()
        return total_seconds

    # printer
    def print_bgn_time(self):
        print('\nSTART: {}\n'.format(self.bgn_time))

    def print_end_time(self):
        print('\nFINISH: {}'.format(self.end_time))

    def print_gap_time(self):
        print('TOTAL: {}\n'.format(self.gap_time))

    def print_all_times(self):
        print('\nSTART: {}'.format(self.bgn_time))
        print('FINISH: {}'.format(self.end_time))
        print('TOTAL: {}\n'.format(self.gap_time))
