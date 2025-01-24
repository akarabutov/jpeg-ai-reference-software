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

import unittest

import os
import re
import sys
import torch
import numpy as np
import json
from typing import List
import tempfile
import subprocess
from src.codec.common import Image

class TestRecoTask(unittest.TestCase):

    def init_images(self):
        self.path_in = 'data/test'
        self.imgs = ['00030_TE_560x888_8bit_sRGB.png']
        self.dvc_pull([os.path.join(self.path_in, img) for img in self.imgs])

    def dvc_pull(self, file_list: List[str]) -> int:
        cmd = [sys.executable, '-m', 'dvc', 'pull'] + [f'{x}.dvc' for x in file_list]
        os.system(' '.join(cmd) + " > /dev/null 2>&1")

    # Test
    def test_rdi(self):
        self.init_images()
        imgs_list = ["--imgs"] +  self.imgs
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_bin_path = os.path.join(tmpdir, "bit.bin")
            cmd_encoder = [sys.executable, "-m", "src.reco.coders.encoder",
                           os.path.join(self.path_in, self.imgs[0]),
                           tmp_bin_path,
                           "-target_bpps", "12",
                           "-rdi.cicp_info_present_flag","1",
                           "-rdi.mdcv_info_present_flag","1",
                           "-rdi.clli_info_present_flag","1",
                           "-rdi.dm_present_flag","1",
                           ]
            enc_ans = subprocess.call(cmd_encoder, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.assertEqual(enc_ans, 0, "Encoder crashed")
            
            cmd_decoder = [sys.executable, "-m", "src.reco.coders.decoder",
                           tmp_bin_path,
                           os.path.join(tmpdir, "tmp.png"),
                           # "--device", "cpu",
                           ]
            dec_ans = subprocess.call(cmd_decoder, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.assertEqual(dec_ans, 0, "Decoder crashed")


if __name__ == "__main__":
    unittest.main()