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
        self.dvc_pull([os.path.join(self.path_in, img) for img in self.imgs], [" > /dev/null"])

    def create_udi_info(self, filename):
        f_size = np.random.randint(10, 200)
        data = np.random.randint(0,255,size=f_size)
        data.tofile(filename)
        
    def compare_files(self, fn1: str, fn2: str):
        data1 = np.fromfile(fn1, dtype=np.int8)
        data2 = np.fromfile(fn2, dtype=np.int8)
        return (data1 == data2).all()
            
    def dvc_pull(self, file_list: List[str], additional_args: List[str] = None) -> int:
        cmd = [sys.executable, '-m', 'dvc', 'pull'] + [f'{x}.dvc' for x in file_list]
        if additional_args is not None:
            cmd += additional_args
        os.system(' '.join(cmd) + " > /dev/null 2>&1")
        
    # Test
    def test_udi(self):
        self.init_images()
        tmp_file_in = tempfile.NamedTemporaryFile("w", delete=False)
        tmp_file_in.close()
        tmp_file_out = tempfile.NamedTemporaryFile("w", delete=False)
        tmp_file_out.close()
        self.create_udi_info(tmp_file_in.name)
        imgs_list = ["--imgs"] +  self.imgs
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_bin_path = os.path.join(tmpdir, "bit.bin")
            cmd_encoder = [sys.executable, "-m", "src.reco.coders.encoder",
                           os.path.join(self.path_in, self.imgs[0]),
                           tmp_bin_path,
                           "-target_bpps", "12",
                           "-udi.filepath", tmp_file_in.name
                           ]
            enc_ans = subprocess.call(cmd_encoder, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.assertEqual(enc_ans, 0, "Encoder crashed")
            
            cmd_decoder = [sys.executable, "-m", "src.reco.coders.decoder",
                           tmp_bin_path,
                           os.path.join(tmpdir, "tmp.png"),
                           # "--device", "cpu",
                           "-udi.filepath", tmp_file_out.name
                           ]
            dec_ans = subprocess.call(cmd_decoder, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.assertEqual(dec_ans, 0, "Decoder crashed")
            
            self.assertTrue(self.compare_files(tmp_file_in.name, tmp_file_out.name), "Files are not the same")
            
            os.remove(tmp_file_in.name)
            os.remove(tmp_file_out.name)


if __name__ == "__main__":
    unittest.main()