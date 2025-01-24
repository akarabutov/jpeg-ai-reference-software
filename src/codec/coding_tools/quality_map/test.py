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
import torch
import torch.nn.functional as F
import numpy as np
import json
from typing import List
import tempfile
import subprocess
from src.codec.common import Image

class TestQMapTask(unittest.TestCase):

    def prepare_images(self, output_dir: str, size: int) -> List[str]:
        ans = list()
        rec_top = size // 4
        rec_bottom = 3 * size // 4
        t = torch.zeros((1,3,size,(size+7)), dtype=torch.float, device=torch.device('cpu'))
        t[:,:,rec_top:rec_bottom, rec_top:rec_bottom] = 1.0
        img = Image.create_from_tensor(t)
        fn = os.path.join(output_dir, 'img.png')
        ans.append(fn)
        img.write_file(fn)
        w = torch.tensor([[[[3., 1., 1.],
                            [1., 0., -1.],
                            [-1., -1., -3.]]]]).repeat(3,1,1,1)
        t2 = F.conv2d(t, w, groups=3, padding=1).abs()
        w = torch.ones( (3,1,7,7), dtype=torch.float)
        t2 = F.conv2d(t2, w, groups=3, padding=3).abs()
        t2.clamp_(0,1)
                
        img2 = Image.create_from_tensor(t2)
        fn = os.path.join(output_dir, 'img_positive.png')
        ans.append(fn)
        img2.write_file(fn)
        
        return ans
        
    
    def run_reco_test(self, img_path, results_path, additional_args=[], run_decoder: bool= True, only_cpu: bool = False):
        from scripts.run_eval_script import run_eval_script
        if only_cpu:
            additional_args += ["--only_cpu",  "--cpu_threads_limit", "1"]        
        run_eval_script(results_path, "cfg/eval/tools_onoff_enc.json", 
                        additional_args=["--in_dir", img_path] + additional_args + ["--calc_encoder_metrics", "0"],
                        verbose=False)
        if run_decoder:
            run_eval_script(results_path, "cfg/eval/tools_onoff_dec.json", 
                            additional_args=["--in_dir", img_path, 
                                             "--force_encdec_match", "1",
                                             "--calc_decoder_metrics", "0"] + additional_args,
                            verbose=False)
        
            for root, dirs, files in os.walk(results_path):
                root_bn = os.path.basename(root)
                if "failed.logs" in files:
                    failed_file_path = os.path.join(root, "failed.logs")
                    file_stats = os.stat(failed_file_path).st_size
                    if file_stats != 0:
                        with open(failed_file_path, "r") as f:
                            print(f"Failed on the following files:\n{f.read()}")
                    self.assertTrue((file_stats == 0) )
                if run_decoder and root_bn=="compare":
                    for fp in files:
                        f_path = os.path.join(root, fp)
                        with open(f_path, "r") as fid:
                            cmp_txt = fid.read()
                            has_mismatch = "MISMATCH" in cmp_txt
                            if has_mismatch:
                                print(f"Comparing failed on: {cmp_txt}")
                            self.assertFalse(has_mismatch)

        
    def calc_mse(self, fn1: str, fn2: str, mask_fn: str) -> float:
        img1 = Image.read_file(fn1, data_range=(0,1))
        img2 = Image.read_file(fn2, data_range=(0,1))
        mask_dn = Image.read_file(mask_fn, data_range=(0,1))
        img1.to_YUV_()
        img2.to_YUV_()
        mse = torch.mean( mask_dn.get_component('a') * (img1.get_component('a')-img2.get_component('a')) ** 2 )
        return mse
            
    ## Tests
    def test_qmap_rect(self):
        mse_dict = dict()
        with tempfile.TemporaryDirectory() as path_in:
            img_files = self.prepare_images(path_in, 400)
            for mask_path in [None, img_files[1]]:
                with tempfile.TemporaryDirectory() as path_out:
                    args = ["-target_bpps", "12", "--imgs", img_files[0]]
                    if mask_path is not None:
                        args += ["--cfg", "./cfg/tools/quality_map.json", "-model.CCS_SGMM.tools_common.qual_map.ROI_map_in_file", mask_path]
                    self.run_reco_test(path_in, path_out, 
                                       additional_args=args,
                                       run_decoder=False,
                                       only_cpu=False)
                    for root, _, files in os.walk(path_out):
                        cur_dir = os.path.basename(root)
                        if cur_dir == "rec":
                            rec_fn = files[0]
                            rel_path = os.path.relpath(root, path_out)
                            mse = self.calc_mse(os.path.join(path_in, img_files[0]), os.path.join(root, rec_fn), img_files[1])
                            if rel_path not in mse_dict:
                                mse_dict[rel_path] = list()
                            mse_dict[rel_path].append(mse)


        for v in mse_dict.values():
            self.assertTrue(v[0] > v[1])

    


if __name__ == "__main__":
    unittest.main()