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


class MonitorBase:
    def __init__(self):
        self.buffer = dict()
        self.table = None
        self.home_dir = str()

    def add_event(self, event):
        raise NotImplementedError

    def init(self, event, *args):
        """Start event that has duration.

        Args:
            event:

        Returns:

        """
        raise NotImplementedError

    def term(self, event, *args):
        """Finish event that has duration.

        Args:
            event:

        Returns:

        """
        raise NotImplementedError

    def save_results(self):
        """Save results to the directory with the filename 'filename.CSV'

        Args:
            filename:

        Returns:

        """
        raise NotImplementedError

    def save_table(self, filename):
        """Save results to the directory with the filename 'filename.CSV'

        Args:
            filename:

        Returns:

        """
        file_path = filename
        if self.home_dir:
            file_path = os.path.join(self.home_dir, filename)

        table = self.table.sort_values(by='Datetime')
        table = table.drop(columns=['Datetime'])
        table.to_csv(file_path, index=False)

    def set_directory(self, root_dir):
        """Set directory for saving data.

        Args:
            root_dir:

        Returns:

        """
        raise NotImplementedError

    @staticmethod
    def record_data(event: str, *args):
        raise NotImplementedError

    @staticmethod
    def set_subdir(root_dir, subdir_name):
        """Set the directory for saving the collector's data.

        Args:
            root_dir:
            subdir_name:

        Returns:

        """
        subdir = os.path.join(root_dir, subdir_name)
        os.makedirs(subdir, exist_ok=True)
        return subdir

    @staticmethod
    def set_time():
        return datetime.now().time()
