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
import sys

class Downloader:
    def __init__(self, models_dir, critical_for_file_absence: bool = False):
        self.models_dir = models_dir
        python_path = sys.executable
        python_bin_dir = os.path.dirname(python_path)
        self.dvc_path = os.path.join(python_bin_dir, 'dvc')
        self.downloaded_models_list = list()
        self.critical_for_file_absence = critical_for_file_absence

    def download_models(self, models_list):
        for model_name in models_list:
            d = os.path.join(self.models_dir, model_name)
            if d in self.downloaded_models_list:
                continue
            if os.path.exists(d):
                dvc_files_list = [os.path.join(d, x) for x in os.listdir(d) if x.endswith('.dvc')]
                if len(dvc_files_list) == 0:
                    print(f'No *.dvc files for model {model_name} in repository!!!')
                    exit(-120)
                else:
                    dvc_files = ' '.join(dvc_files_list)
                    os.system(f'{self.dvc_path} fetch {dvc_files}')
                    os.system(f'{self.dvc_path} checkout -f {dvc_files}')
                    self.downloaded_models_list.append(d)
            else:
                print(f'No information about model {model_name} in repository!!!')
                exit(-100)

    def get_file_path(self, model_name, file_name):
        path = os.path.join(self.models_dir, model_name, file_name)
        if not os.path.exists(path):
            if self.critical_for_file_absence:
                print(f'CRITICAL: cannot find file {file_name} in model {model_name}')
                exit(-200)
            else:
                path = None
        return path
