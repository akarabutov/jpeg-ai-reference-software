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
import sys
import numpy as np
import json
from typing import List, Tuple
import tempfile
import subprocess
import commentjson
from src.codec.common import Image
from src.codec import CTC_get_default_fn

class TestRecoTask(unittest.TestCase):
    IMG_COUNT = 1

    def init_images(self):
        self.path_in = 'data/test'
        self.imgs = ['00030_TE_560x888_8bit_sRGB.png', '00018_TE_3032x1856_8bit_sRGB.png']
        self.dvc_pull([os.path.join(self.path_in, img) for img in self.imgs])

    def init_images_tiles(self):
        self.path_in = 'data/test'
        self.imgs = ['00001_TE_2096x1400_8bit_sRGB.png']
        self.dvc_pull([os.path.join(self.path_in, img) for img in self.imgs])

    def create_png_file(self, dir_path: str, index: int, max_size: int):
        w = np.random.randint(100, max_size//2)*2
        h = np.random.randint(100, max_size//2)*2
        rgb_data = torch.randint(0, 255, (1,3,h,w))
        img = Image(w,h, [0,255], data=rgb_data)
        img.write_file(os.path.join(dir_path, "{:05d}_TE_{}x{}_8bit_sRGB.png".format(index, w, h)))
        return w, h
        
    def create_yuv_file(self, dir_path: str, index: int, max_size: int, yuv_format: str, bit_depth:int=8):
        w = np.random.randint(100, max_size//2)*2
        h = np.random.randint(100, max_size//2)*2
        max_value = (1<<bit_depth) - 1
        luma_data = torch.randint(0, max_value, (1,1,h,w), dtype=torch.float)
        h2,w2 = Image.calc_chroma_size_444_to_420((h,w)) if yuv_format == '420' else (h,w)
        chroma_data = torch.randint(0, max_value, (1,2,h2,w2), dtype=torch.float)
        img = Image.create_from_tensors(luma_data, chroma_data[:,0:1], chroma_data[:,1:2],
                                        [0,max_value], bit_depth=bit_depth, color_space='yuv', format=yuv_format)
        img.write_file(os.path.join(dir_path, "{:05d}_TE_{}x{}_{}bit_YUV{}.yuv".format(index, w, h, bit_depth, yuv_format)))
        return w, h

    def convert_png2yuv(self, png_path:str, yuv_base_path:str, yuv_name_prefix:str, yuv_format: str = '444', yuv_bd: int = 8) -> str:
        img = Image.read_file(png_path)
        img.to_YUV_()
        img.to_format_(yuv_format)
        h,w = img.shape[-2:]
        yuv_name = f"{yuv_name_prefix}_{w}x{h}_YUV{yuv_format}_{yuv_bd}bit.yuv"
        yuv_path = os.path.join(yuv_base_path, yuv_name)
        img.write_yuv(yuv_path, yuv_bd)
        return yuv_name
        
    def calc_mse(self, fn1: str, fn2: str, update_img_fn = None) -> float:
        img1 = Image.read_file(fn1, data_range=(0,1))
        img2 = Image.read_file(fn2, data_range=(0,1))
        img1.to_YUV_()
        img2.to_YUV_()
        img1_a = img1.get_component('a')
        img2_a = img2.get_component('a')
        if update_img_fn:
            img1_a = update_img_fn(img1_a)
            img2_a = update_img_fn(img2_a)
        mse = torch.mean( (img1_a-img2_a) ** 2 )
        return mse
        
    def run_reco_test_short(self, img_path, results_path, additional_args:List[str]=list()) -> Tuple[str, str]:
        # Run encoder
        bin_path = os.path.join(results_path, "bin.bin")
        args = ["--cfg", "cfg/tools_off.json", "cfg/profiles/base.json"] + additional_args
        enc_ans = self.run_encoder(img_path, bin_path, args)
        self.assertEqual(enc_ans, 0, r"Encoder crashed")
        
        # Run decoder
        img_output = os.path.join(results_path, "out.png")
        dec_ans = self.run_decoder(bin_path, img_output)
        self.assertEqual(dec_ans, 0, r"Decoder crashed")
        
        return bin_path, img_output
            

    def run_reco_test_full(self, img_path, results_path, additional_args=[], run_decoder: bool= True, only_cpu: bool = False):
        from scripts.run_eval_script import run_eval_script
        additional_args += ["--cpu_threads_limit", "1"]
        if only_cpu:
            additional_args += ["--only_cpu"]        
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

    def run_recotask_yuv(self, fmt='420'):
        self.init_images()
        with tempfile.TemporaryDirectory() as path_in, tempfile.TemporaryDirectory() as path_out:
            files_list = list()
            cur_idx = 0
            for bits in [8, 10]:
                for img in self.imgs:
                    a = self.convert_png2yuv(os.path.join(self.path_in, img), path_in, f"{cur_idx:05}_TE", fmt, bits)
                    cur_idx += 1
                    files_list.append(a)
            self.run_reco_test_full(img_path=path_in, 
                               results_path=path_out, 
                               additional_args=["--use_yuv", "1", "--imgs"] + files_list)
            
    def dvc_pull(self, file_list: List[str]) -> int:
        cmd = [sys.executable, '-m', 'dvc', 'pull'] + [f'{x}.dvc' for x in file_list]
        os.system(' '.join(cmd) + " > /dev/null 2>&1")        
            
    ## Tests
    def test_recotask_yuv420(self):
        self.run_recotask_yuv('420')
            
    def test_recotask_yuv422(self):
        self.run_recotask_yuv('422')

    def test_recotask_yuv444(self):
        self.run_recotask_yuv('444')
        
    def test_brm(self):
        # temporary removed this test
        self.init_images()
        files_s = dict()
        r = re.compile(r'(?P<idx>\d+)_TE_(?P<w>\d+)x(?P<h>\d+)')
        for img in self.imgs:
            g = r.search(img)
            idx = int(g.group('idx'))
            w = int(g.group('w'))
            h = int(g.group('h'))
            files_s[idx] = w*h
        with tempfile.TemporaryDirectory() as path_out:
                
            self.run_reco_test_full(img_path=self.path_in, 
                               results_path=path_out, 
                               additional_args=["--cfg", "./cfg/BRM/regen_list.json",
                               "-model.bitrate_matcher.bitrate_config_name", "tmp.txt",
                               "--imgs"] + self.imgs, run_decoder=False)
            for op_name in os.listdir(path_out):
                op_path = os.path.join(path_out, op_name)
                if os.path.isdir(op_path):
                    for task_name in os.listdir(op_path):
                        cur_output_dir = os.path.join(op_path, task_name)
                        if os.path.isdir(cur_output_dir):
                            with open(os.path.join(cur_output_dir, "cfg.json"), 'r') as f:
                                cfg = json.load(f)
                            target_bpps = cfg['target_bpps']
                            bit_dir_path = os.path.join(cur_output_dir, 'bit')
                            bit_fl = [x for x in os.listdir(bit_dir_path)]
                            for idx,s in files_s.items():
                                for bpp in target_bpps:
                                    name = "VM_{:05d}_TE_{:03d}.bits".format(idx, bpp)
                                    self.assertTrue(name in bit_fl)
                                    f_size = os.stat(os.path.join(bit_dir_path, name)).st_size
                                    cur_bpp = f_size * 8 / s
                                    self.assertTrue(cur_bpp < (bpp * 0.01 * 1.1))                      
                                
        
    def run_recotask_png(self, additional_args:List[str]=list()):
        self.init_images()
        with tempfile.TemporaryDirectory() as path_out:
            self.run_reco_test_short(os.path.join(self.path_in, self.imgs[0]),
                                     results_path=path_out,
                                     additional_args=additional_args)

    def test_recotask_png444_dependent_tiles(self):
        self.init_images_tiles()
        mses = list()
        in_img_path = os.path.join(self.path_in, self.imgs[0])
        additional_args = ["--cfg", "./cfg/tools_on.json"]
        with tempfile.TemporaryDirectory() as path_out:
            _, img_path = self.run_reco_test_short(in_img_path,
                            results_path=path_out,
                            additional_args=additional_args)
            mses.append(self.calc_mse(in_img_path, img_path))
        for use_threads in [0,1]:
            with tempfile.TemporaryDirectory() as path_out:
                if not use_threads:
                    additional_args += ["--cfg", "./cfg/tools/DependentRegions.json"]
                if use_threads:
                    additional_args += ["./cfg/tools/ECThread8.json"]
                tmp_bin_path, img_path = self.run_reco_test_short(in_img_path,
                               results_path=path_out,
                               additional_args=additional_args)
                # Get stream structure
                str_str = self.get_stream_structure(tmp_bin_path)
                for n in ["r_prim_substream", "r_sec_substream"]:
                    self.assertIn(n, str_str)
                    self.assertIn("region_0", str_str[n])
                    self.assertIn("Parameters", str_str[n]["region_0"])
                    self.assertIn("Subregions", str_str[n]["region_0"]["Parameters"])
                    self.assertGreater(len(str_str[n]["region_0"]["Parameters"]["Subregions"]), 1)
                    
                mses.append(self.calc_mse(in_img_path, img_path))
        self.assertLess((mses[-1]-mses[-2]).abs(), 1E-4, "Multi-thread and single thread versions have noticable difference in quality")
        self.assertLess((mses[0]-mses[-2]).abs(), 1E-4, "Single thread versions with dependent tiles have noticable difference in quality compared to a version without dependent tiles")
        self.assertLess((mses[0]-mses[-1]).abs(), 1E-4, "Multi-thread versions with dependent tiles have noticable difference in quality compared to a version without dependent tiles")
        

    def test_recotask_png444(self):
        self.run_recotask_png()

    def test_recotask_png420(self):
        self.run_recotask_png(["-c_ver_value", "2",
                               "-c_hor_value", "2"])
        
    def run_encoder(self, input_image_fn: str, output_bin_path: str, additional_args: List[str]) -> int:
        cmd_encoder = [sys.executable, "-m", "src.reco.coders.encoder",
                    input_image_fn,
                    output_bin_path,
                    "-target_bpps", "12",
                    #"-target_device", "gpu"
                    ] + additional_args
        # , stdout=subprocess.STDOUT, env=my_env
        enc_ans = subprocess.call(cmd_encoder, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return enc_ans
    
    def run_decoder(self, input_bin_path: str, output_reco_fn: str, additional_args: List[str] = list()) -> int:
        cmd_decoder = [sys.executable, "-m", "src.reco.coders.decoder",
                    input_bin_path,
                    output_reco_fn,
                    #"--device", "cpu",
                    #"-target_device", "cpu"
                    ] + additional_args 
        dec_ans = subprocess.call(cmd_decoder, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return dec_ans
    
    def get_stream_structure(self, filename) -> dict:
        tmpfn = tempfile.NamedTemporaryFile(delete=False)
        tmpfn.close()
        cmd_probe = [sys.executable, "-m", "scripts.bitstream_probe",
                    filename,
                    "--silent",
                    "--json_output", tmpfn.name
                    ] 
        probe_ans = subprocess.call(cmd_probe, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(tmpfn.name, 'r') as f:
            ans = commentjson.load(f)
        
        self.assertEqual(probe_ans, 0, f"Cannot probe a file {filename}")
        os.remove(tmpfn.name)
        
        return ans
    
    def test_threads(self):
        self.init_images()
        additional_args=["./cfg/tools/IndependentRegions.json",
                         "./cfg/tools/ECThread8.json"]
        log_threads_count = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_bin_path, _ = self.run_reco_test_short(os.path.join(self.path_in, self.imgs[0]),
                                                       tmpdir,
                                                       additional_args)
            # Get stream structure
            str_str = self.get_stream_structure(tmp_bin_path)
            self.assertIn('picture_header', str_str)
            self.assertIn('region_0', str_str['picture_header'])
            self.assertIn('Flags', str_str['picture_header']['region_0'])
            self.assertIn('multi_threading_z', str_str['picture_header']['region_0']['Flags'])
            self.assertEqual(1, str_str['picture_header']['region_0']['Flags']['multi_threading_z'])
            self.assertIn('log2_num_threads_z_minus1', str_str['picture_header']['region_0']['Flags'])
            self.assertEqual(log_threads_count, str_str['picture_header']['region_0']['Flags']['log2_num_threads_z_minus1'])
        
            self.assertIn('tool_header', str_str)
            self.assertIn('region_0', str_str['picture_header'])
            self.assertIn('Flags', str_str['picture_header']['region_0'])
            for i in range(2):
                name = f'multi_threading_r[{i}]'
                self.assertIn(name, str_str['picture_header']['region_0']['Flags'])
                self.assertEqual(1, str_str['picture_header']['region_0']['Flags'][name])
                name = f'log2_num_threads_r_minus1[{i}]'
                self.assertIn(name, str_str['picture_header']['region_0']['Flags'])
                self.assertEqual(log_threads_count, str_str['picture_header']['region_0']['Flags'][name])
            

        
    def test_independent_tiles(self):
        #self.init_images()
        self.init_images_tiles()
        region_mse = list()
        additional_args=["--cfg", 'cfg/tools_on.json', 'cfg/profiles/base.json']
        reduction_fn = lambda x: x[..., 1024:, :]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_fn = os.path.join(self.path_in, self.imgs[0])
            _, rec_path = self.run_reco_test_short(input_fn, tmpdir, additional_args)
            region_mse.append(self.calc_mse(input_fn, rec_path, reduction_fn))
            
        for use_threads in [0,1]:
            with tempfile.TemporaryDirectory() as tmpdir:
                threads_number = 8 if use_threads else 1
                tmp_bin_path = os.path.join(tmpdir, "bit.bin")
                input_fn = os.path.join(self.path_in, self.imgs[0])
                additional_args2 = additional_args
                additional_args2 += ["./cfg/tools/IndependentRegions.json"] 
                if use_threads:
                    additional_args2.append("./cfg/tools/ECThread8.json")
                # Encode a stream with support of independent substreams
                enc_ans = self.run_encoder(input_fn, tmp_bin_path, additional_args=additional_args2)
                self.assertEqual(enc_ans, 0, f"Encoder crashed. Threads number is {threads_number}")

                tmp_bin2_path = os.path.join(tmpdir, "bit2.bin")
                # Extract only one of the substreams
                ext_ans = os.system(f'{sys.executable} -m scripts.bitstream_extractor {tmp_bin_path} {tmp_bin2_path} --remove_resi_substreams 0')
                self.assertEqual(ext_ans, 0, "Cannot extract substream")

                tmp_rec_output_path = os.path.join(tmpdir, "out2.png")
                # Check decodability of the new stream
                dec_ans = self.run_decoder(tmp_bin2_path, tmp_rec_output_path)
                self.assertEqual(dec_ans, 0, f"Decoder crashed. Threads number is {threads_number}")
                region_mse.append(self.calc_mse(input_fn, tmp_rec_output_path, reduction_fn))
                
                # Check structure of the new stream
                stream_data = self.get_stream_structure(tmp_bin2_path)
                
                self.assertIn("r_prim_substream", stream_data, f"Stream doesn't have primary substream. Threads number is {threads_number}")
                self.assertNotIn("region_0", stream_data['r_prim_substream'], f"Region 0 is presented in primary substream. Threads number is {threads_number}")
                self.assertIn("r_sec_substream", stream_data, f"Stream doesn't have primary substream. Threads number is {threads_number}")
                self.assertNotIn("region_0", stream_data['r_sec_substream'], f"Region 0 is presented in secondary substream. Threads number is {threads_number}")
                
        self.assertLess((region_mse[-1]-region_mse[-2]).abs(), 1E-4, "Quality of results with and without threads are different")
        for case_id in range(2):
            idx = case_id + 1
            self.assertLess((region_mse[0]-region_mse[idx]).abs(), region_mse[0]*2, f"Reconstructed part of image in a case of {case_id} is too bad")
                
            
        
    def test_progressive_decoding(self):
        self.init_images()
        for num_threads in [1,2,4,8,16]:
            additional_args = ["-model.CCS_SGMM.tools_common.qual_map.num_threads", f"{num_threads}",
            		"-model.CCS_SGMM.tools_common.num_threads_z", f"{num_threads}",
		            "-model.CCS_SGMM.tools_common.model_common.num_threads_r", f"{num_threads}",
            ]
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_bin_path = os.path.join(tmpdir, "bit.bin")
                input_fn = os.path.join(self.path_in, self.imgs[0])
                enc_ans = self.run_encoder(input_fn, tmp_bin_path, additional_args)
                self.assertEqual(enc_ans, 0, "Encoder crashed")
                last_mse = None
                
                for i, (chs_Y, chs_UV) in enumerate(((0, 16), (32, 32), (96, 48))):
                    additional_args = ["-model.CCS_SGMM.tools_common.model_y.common_modules.num_decode_chs", str(chs_Y),
                                    "-model.CCS_SGMM.tools_common.model_uv.common_modules.num_decode_chs", str(chs_UV)]
                    output_fn = os.path.join(tmpdir, f"tmp_{i}.png")
                    dec_ans = self.run_decoder(tmp_bin_path, output_fn, additional_args)
                    self.assertEqual(dec_ans, 0, "Decoder crashed")
                    cur_mse = self.calc_mse(input_fn, output_fn)
                    if last_mse is not None:
                        if last_mse < cur_mse:
                            print(tmpdir)
                        self.assertGreaterEqual(last_mse, cur_mse, msg=f"Num threads is {num_threads}, channels number is ({chs_Y}, {chs_UV}). Output directory is {tmpdir}")
                    last_mse = cur_mse
               



if __name__ == "__main__":
    unittest.main()
