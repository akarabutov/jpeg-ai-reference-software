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
import sys
import torch
import numpy as np
import subprocess
import shutil
import tempfile
import commentjson
from typing import List
from src.codec.common import Image
from src.codec import get_cfg_def_dir

class TestTrainTask(unittest.TestCase):
    IMG_COUNT = 2


    def create_png_file(self, dir_path: str, index: int, max_size: int):
        w = np.random.randint(200, max_size)
        h = np.random.randint(200, max_size)
        rgb_data = torch.randint(0, 255, (1,3,h,w))
        img = Image(w,h, [0,255], data=rgb_data)
        fn = "{:05d}_TE_{}x{}_8bit_sRGB.png".format(index, w, h)
        img.write_file(os.path.join(dir_path, fn))
        return fn
        
          
    def dvc_pull(self, file_list: List[str]) -> int:
        cmd = [sys.executable, '-m', 'dvc', 'pull'] + [f'{x}.dvc' for x in file_list]
        #return subprocess.check_output(cmd) #, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, )    
        os.system(' '.join(cmd) + " > /dev/null 2>&1")
        
    def run_train(self, img_path, img_lst, out_dir, additional_args=list()):
        ## Prepare train.json
        with open(os.path.join(get_cfg_def_dir(), "train.json"), "r") as f:
            cfg = commentjson.load(f)
        kk = list(cfg['beta_2_gpus'].keys())
        current_beta = kk[0]
        current_beta_tmp = '-1'
        if torch.cuda.is_available() and len(os.environ['CUDA_VISIBLE_DEVICES'])>0:
            current_beta_tmp = os.environ['CUDA_VISIBLE_DEVICES']
        cfg['beta_2_gpus'] = { current_beta: current_beta_tmp  }
        train_json_path = os.path.join(img_path, "train.json")
        with open(train_json_path, "w") as f:
            commentjson.dump(cfg, f)
        train_cfg = cfg

        ## Prepare stages
        with open(os.path.join(get_cfg_def_dir(), "train_stages.json"), "r") as f:
            cfg = f.read()
        cfg = cfg.replace(']', ', "--epochs", 1]')
        stages_json_path = os.path.join(img_path, "train_stages.json")
        with open(stages_json_path, "w") as f:
            f.write(cfg)
            
        ## Store a list of filesnames
        img_lst_fn = os.path.join(img_path, "lst.txt")
        with open(img_lst_fn, "w") as f:
            f.write('\n'.join(img_lst))
            
        ## Run training
        cmd = [
            sys.executable,
            "-m", "scripts.acc_train_scripts.acc_train_local",
            "--data_dir",  img_path,
            "--lst", img_lst_fn,
            "--val_data_dir", img_path,
            "--val_lst", img_lst_fn, 
            "--train_cfg_json", train_json_path,
            "--train_stages_json", stages_json_path,
            "--train_url", out_dir,
            "--test_data_dir", img_path,
            "--use_automatic_testing", "1",
            "--generate_test_summary", "0",
            "--base_warmup_epoch", "1.0",
            "--use_automatic_testing_best", "0"
        ]
        cmd += additional_args
        try:
            err_code = subprocess.check_call(cmd, stdout=subprocess.PIPE)
        except subprocess.CalledProcessError as inst:
            err_code = inst.returncode
            
        logs_archive_path = os.path.join(os.getcwd(), "train_logs.tgz")
            
        os.system(f"tar -czf {logs_archive_path} {out_dir}/log/ {out_dir}/*/*/val_results.json")
            
        self.assertTrue(err_code == 0)
        
        # Check existance of a file with results of validation
        for stage in train_cfg.get('stages', ['A']):
            if not stage.endswith('_Collection'):
                dir_path = os.path.join(out_dir, stage, current_beta)
                is_ok = os.path.exists(os.path.join(dir_path, "val_results.json"))
                self.assertTrue(is_ok, msg=f"Cannot find json file with results of phase {stage} in {dir_path}. It contains: {os.listdir(dir_path)}")                   
            
        # Check results of tests
        for root, dirs, files in os.walk(out_dir):
            root_root_bn = os.path.basename(os.path.dirname(root))
            if root_root_bn.endswith('_test'):
                if "failed.logs" in files:
                    failed_file_path = os.path.join(root, "failed.logs")
                    file_stats = os.stat(failed_file_path).st_size
                    if file_stats != 0:
                        with open(failed_file_path, "r") as f:
                            print(f"Failed on the following files:\n{f.read()}")
                    self.assertTrue((file_stats == 0) )
                self.assertTrue('summary.txt' in files)
                with open(os.path.join(root, 'summary.txt'), 'r') as f:
                    for l in f:
                        s_arr = l.split('\t')
                        for i in range(13):
                            self.assertNotEqual(s_arr[i], r'None', r"Failed one of the metric")
                        self.assertNotEqual(s_arr[-1], r'None', r"Failed one of the metric")
               
        # Remove file if the training went good         
        os.remove(logs_archive_path)
            
    def test_train(self):
        # Download models 
        init_train_stage = "MSE_VariableRate_12"
        path_to_models=os.path.join(os.getcwd(), "models", "VM_common", "train_stages")
        dvc_lst = [os.path.join(path_to_models, init_train_stage, x, "best.pth") for x in os.listdir(os.path.join(path_to_models, init_train_stage))]
        ans = self.dvc_pull(dvc_lst)
        # Create dummy images
        with tempfile.TemporaryDirectory() as img_path, tempfile.TemporaryDirectory() as out_dir:
            img_lst = list()
            for i in range(self.IMG_COUNT):
                img_lst.append(self.create_png_file(img_path, i, 1512))
            self.run_train(img_path=img_path, 
                               img_lst=img_lst,
                               out_dir=out_dir,
                               additional_args=[
                                    "--copy_to_train_url_dir", path_to_models,
                                    "--resume_from_stage", init_train_stage,
                                    "--opt_type", "adam",
                                    "--amp", "0"
                               ])


if __name__ == "__main__":
    unittest.main()