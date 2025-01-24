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

import re
import torch
import argparse
import json
import os
import shutil
import sys
from typing import Tuple, Dict, List
##
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor as Pool
import torch.nn.functional as F
from multiprocessing import current_process
from typing import List

from src.codec import (CTC_get_default_fn, create_dir_structure, get_cfg_def_dir, get_codec_name,
                       get_codec_version, get_downloader, set_param_recurrent)
from src.codec.coding_tools.coding_engine import CodingEngine
from src.codec.common import ArgParserDecorator as ParserDecorator, Decisions
from src.codec.common.timeslot import Timeslot
from src.codec.common.utils import (config_gpu_list, copy_eval_scripts, execute_eval, logging_str,
                                    start_and_log_str)
from src.codec.datasets import ImageDataset
from src.codec.entropy_coding import ECFactory
from src.codec.utils import cmd_params_loading, ExtendAction
from src.codec.metrics import MetricsProcessor


class LoggerWithOutput(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.err_term = sys.stderr
        self.log = None
        sys.stdout = self
        sys.stderr = self
        
    def open(self, filename):
        if self.log is not None:
            self.log.close()
        self.log = open(filename, "w")
        
    def flush(self):
        self.log.flush()
        self.terminal.flush()
        
    def write(self, message):
        self.terminal.write(message)
        if self.log is not None:
            self.log.write(message)
            self.log.flush()
        
    def close(self):
        if self.log is not None:
            self.log.close()
            self.log = None
        
    def finish(self):
        sys.stdout = self.terminal    
        sys.stderr = self.err_term   
    
    def __del__(self):
        self.close()
        self.finish()


# ######################################################################################################################
#  Parameter methods
# ######################################################################################################################
def def_eval_base_parser():
    prog = 'Codec: name={}, version={}'.format(get_codec_name(), get_codec_version())
    this = argparse.ArgumentParser(prog, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    this.add_argument('--cfg',
                      default=list(),
                      nargs='+',
                      type=str,
                      action=ExtendAction,
                      help='path to config file(s)')

    this.add_argument('--gpu_greedy',
                      default=False,
                      action='store_true',
                      help=r'use gpus even if they are already in use')


    this.add_argument('--rebuild_ae_cache',
                      default=1,
                      choices=[0,1],
                      type=int,
                      help=r'Regenerate cache of ANS')

    this.add_argument('--gpu_max', type=int, help='maximum number of gpus to use')

    this.add_argument('--gpu_ids', type=str, default=None, help='maximum number of gpus to use')

    this.add_argument('--coding_type',
                      type=str,
                      default='enc_dec',
                      choices=['enc', 'dec', 'enc_dec'],
                      help='Type of simulation: "enc" is only Encoder, "dec" is only Decoder, '
                      '"enc_dec" is encoder and decoder (default)')

    this.add_argument(
        '--force_encdec_match',
        type=int,
        default=0,
        choices=[0, 1],
        help='Force matching of enc/dec results even if you set coding_type to enc or dec')

    this.add_argument('--resume_eval',
                      default=False,
                      action="store_true",
                      help='Continue simulation and generate only absent streams')

    this.add_argument('--store_rec',
                      type=int,
                      default=1,
                      help='Store PNG files (rec and ori directories)')

    this.add_argument('--store_bit', type=int, default=1, help='Store BIT files (bit directory)')

    this.add_argument('--imgs', type=str, nargs="+", default=None, help='List images for processing')
    
    this.add_argument('--overwrite', dest='overwrite', action='store_true')
    
    this.add_argument('--skip_loading_error', 
                      default=False,
                      action="store_true",
                      help=r"Skip error of models loading if it happends")    

    this.add_argument('--models_dir_name',
                      type=str,
                      default="models",
                      help='Name of a directory with models')
    
    # CPU only options
    this.add_argument('--only_cpu',
                      default=False,
                      action='store_true',
                      help='Perform scripts only on CPU')

    this.add_argument('--cpu_threads_limit',
                      type=int,
                      default=-1,
                      help=r'Maximum number of used threads '
                      r'(works only with --only_cpu option)')

    this.add_argument('--in_dir', type=str, help='input directory', default='data/test')
    this.add_argument('--out_dir', type=str, help='output directory', default='results/test')

    this.add_argument('--calc_encoder_metrics',
                      help=r'Calculate encoder metrics',
                      type=int,
                      default=1,
                      choices=[0,1])

    this.add_argument('--calc_decoder_metrics',
                      help=r'Calculate decoder metrics',
                      type=int,
                      default=0,
                      choices=[0,1])

    this.add_argument('--no_per_ratepoint_config',
                      default=1,
                      type=int,
                      choices=[0, 1],
                      help=r'Use only base configuration without usage configuration from '
                      r'per-image-per-bpp directory')
    this.add_argument('--only_base_config',
                      default=False,
                      action='store_true',
                      help=r'Use only base configuration without usage configuration '
                      r'from per-image directory')
    
    this.add_argument('--use_qual_map',
                      help=r'Use quality map',
                      type=int,
                      default=0,
                      choices=[0,1])    
    
    this.add_argument('--use_yuv',
                      type=int,
                      default=0,
                      help=r'Use YUV input instead of png')

    return this


def def_eval_parser_decorator(base_parser: ArgumentParser):
    this = ParserDecorator(base_parser)

    return this


class CodecEval:
    def __init__(self, task_name: str, base_parser, parser_decorator):
        self.task_name = task_name
        self.base_parser = base_parser
        self.parser_decorator = parser_decorator
        self.ce = CodingEngine()
        self.codec_name = get_codec_name()
        self.codec_version = get_codec_version()


    def print_coder_info(self):
        print('Codec/Eval {}: name={}, version={}'.format(self.task_name, self.codec_name,
                                                          self.codec_version))
        
    def init_metric_proc(self, cmd_args: List[str] = None, cmd_args_add = True):
        if cmd_args is None:
            cmd_args = sys.argv
        metric_proc = MetricsProcessor()
        ap = self.base_parser
        if cmd_args_add:
            metric_proc.add_arguments(ap)
        metrics_args = metric_proc.parse_arguments(ap, cmd_args)
        self.metrics_args = list()
        metrics_args_dict = vars(metrics_args)
        for k,v in metrics_args_dict.items():
            s = f"--{k}"
            if s in cmd_args:
                idx = cmd_args.index(s) 
                if idx != -1:
                    self.metrics_args.append(cmd_args[idx])
                    if isinstance(v, list):
                        self.metrics_args += v
                    else:
                        self.metrics_args += [v]
        self.metric_proc = metric_proc        


    @staticmethod
    def init_cuda():
        y = torch.ones((1, 1, 1, 1), device='cuda')
        z = F.conv2d(y, y)  # noqa: F841
        
    @staticmethod
    def config_bit_file(bit_dir, img_fn, bpp_int: int):
        from .. import get_codec_name, get_correct_bit_name
        img_bn = os.path.splitext(os.path.basename(img_fn))[0]
        bit_name = get_correct_bit_name(get_codec_name(), img_bn, bpp_int)
        bit_fpath = os.path.join(bit_dir, bit_name + '.bits')
        return bit_name, bit_fpath

    @staticmethod
    def config_rec_image(rec_dir: str, input_img_fn: str, bpp_val: int) -> Tuple[str, str]:
        """It generates a name and path to the reconstructed image

        Args:
            rec_dir (str): path of an output directory
            input_img_fn (str): a name of the input image
            bpp_val (int): value of bpp (*100)

        Returns:
            Tuple[str, str]: name of the output file and its path
        """
        from .. import get_codec_name
        ii_bn, ii_ext = os.path.splitext(input_img_fn)
        rec_file = f'{get_codec_name()}_{ii_bn}_{bpp_val:03d}{ii_ext}'
        rec_path = os.path.join(rec_dir, rec_file)        
        return rec_file, rec_path

    # ##################################################################################################################
    #  Parameter methods
    # ##################################################################################################################
    def update_kwargs_params(self, kwargs, params_preprocess=None):
        #kwargs, unknown_params = self.base_parser.parse_known_args()
        kwargs = vars(kwargs)
        params = cmd_params_loading(self.parser_decorator,
                                    self.ce,
                                    CTC_get_default_fn() if len(kwargs['cfg'])==0 else kwargs['cfg'],
                                    params_preprocess=params_preprocess)

        params = self.ce.store_attrs2dict_recursively()
        return kwargs, params #, unknown_params

    # ##################################################################################################################
    #  init methods
    # ##################################################################################################################
    def init_ce(self, ce: CodingEngine):
        if ce is not None:
            self.ce = ce
        profilers = self.ce.get_profilers()
        self.ce.EC = ECFactory.create_instance(profilers=profilers)


    @staticmethod
    def compute(args: Dict):
        import sys
        python_path = sys.executable
        compute_stage = ""

        input_img_path = args.get('img_path')
        input_img_bn = os.path.basename(input_img_path)
        input_img_fn, input_img_ext = os.path.splitext(input_img_bn)

        bpp_val = args.get('bpp_val')
        gpu_ids = args.get('gpu_list', list())
        ce_args = args.get('ce_args', list())
        cfg = args.get('cfg_output', '')
        kwargs = args.get('kwargs', dict())
        coding_type = kwargs.get('coding_type', 'enc_dec')
        force_encdec_match = kwargs.get('force_encdec_match', 0)
        out_dir = kwargs.get('out_dir', '')
        in_dir = kwargs.get('in_dir', '')
        calc_decoder_metrics = kwargs.get('calc_decoder_metrics', 0)
        calc_encoder_metrics = kwargs.get('calc_encoder_metrics', 0)
        metrics_args = kwargs.get('metrics_args', [])
        resume_eval = kwargs.get('resume_eval', False)
        models_dir_name = kwargs.get('models_dir_name', 'models')
        skip_loading_error = kwargs.get('skip_loading_error', False)

        set_cmd_for_enc = args['set_cmd_for_enc']
        set_cmd_for_dec = args['set_cmd_for_dec']
        set_cmd_for_cmp = args['set_cmd_for_cmp']
        
        encoder_inst = kwargs.get('encoder_inst', None)
        decoder_inst = kwargs.get('decoder_inst', None)
        compare_inst = kwargs.get('compare_inst', None)
        
        use_qual_map = kwargs.get('use_qual_map', 0)
        fp = LoggerWithOutput()        
        try:
            gpu_id_idx = 0
            if 'MainProcess' not in current_process().name:
                gpu_id_idx = int(current_process().name.split('-')[1]) -1
            gpu_id = gpu_ids[gpu_id_idx]
            env=None
            if gpu_id >= 0:
                env = dict()
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                
                
            print(f'Start processing of {os.path.basename(input_img_path)} for bpp {bpp_val * 0.01}')
            
            enc_finished_successfully = True
            dec_finished_successfully = True

            bit_dir = os.path.join(out_dir, 'bit')
            os.makedirs(bit_dir, exist_ok=True)
            _, bit_path = CodecEval.config_bit_file(bit_dir, input_img_fn, bpp_val)
            
            run_enc = coding_type == 'enc' or coding_type == 'enc_dec'
            run_dec = coding_type == 'dec' or coding_type == 'enc_dec'
            enc_log_path = os.path.join(out_dir, 'log', 'enc', '{}_{:03d}.txt'.format(input_img_fn, bpp_val))
            dec_log_path = os.path.join(out_dir, 'log', 'dec', f'{input_img_fn}_{bpp_val:03d}.txt')

            
            
            input_path = os.path.join(in_dir, input_img_bn)
            enc_rec_dir = os.path.join(out_dir, 'rec')
            _, enc_rec_path = CodecEval.config_rec_image(enc_rec_dir, input_img_bn, bpp_val)
            os.makedirs(enc_rec_dir, exist_ok=True)
            dec_rec_dir = os.path.join(out_dir, 'rec_dec')
            dec_rec_file, dec_rec_path = CodecEval.config_rec_image(dec_rec_dir, input_img_bn, bpp_val)
            os.makedirs(dec_rec_dir, exist_ok=True)

            if resume_eval:
                enc_cond = os.path.exists(bit_path) and os.stat(bit_path).st_size != 0
                dec_cond = os.path.exists(dec_rec_path)
                
                if run_enc and enc_cond and run_dec and dec_cond:
                    run_dec = False
                    run_enc = False
                    enc_finished_successfully = False
                    dec_finished_successfully = False                    
                elif (run_enc and enc_cond):
                    run_enc = False
                elif (run_dec and dec_cond):
                    run_dec = False
                    

            if run_enc:
                compute_stage = "encoder"

                fp.open(enc_log_path)
                cmd, cmd_args = set_cmd_for_enc(python_path, input_path, bit_path, enc_rec_path, bpp_val, **kwargs)

                if isinstance(cfg, list):
                    cmd_args += cfg
                else:
                    cmd_args.append(cfg)
                
                cmd_args.append('--models_dir_name')
                cmd_args.append(models_dir_name)
                if skip_loading_error:
                    cmd_args.append('--skip_loading_error')
                cmd_args = cmd_args + ce_args
                
                if use_qual_map:
                    in_mask_path = os.path.join(os.path.dirname(input_path) + "_mask", os.path.basename(input_path))
                    out_mask_path = os.path.join(os.path.dirname(input_path) + "_mask_out", os.path.basename(input_path))
                    cmd_args.append(f"-model.CCS_SGMM.tools_common.model_common.quantizer.qual_map.ROI_map_in_file")
                    cmd_args.append(in_mask_path)
                    cmd_args.append(f"-model.CCS_SGMM.tools_common.model_common.quantizer.qual_map.ROI_map_out_file")
                    cmd_args.append(out_mask_path)

                if calc_encoder_metrics:
                    cmd_args.append("--calc_metrics")
                    cmd_args += metrics_args
                #cmd_args += additional_cmd_args

                device = (f'GPU {gpu_id}' if gpu_id >= 0 else 'CPU')
                print('Start encoder on {} : {}\n'.format(device, ' '.join(cmd+cmd_args)))
                timeslot_dump = Timeslot()    
                timeslot_dump.set_bgn_time()
                if encoder_inst is None:
                    enc_return_code = start_and_log_str(cmd+cmd_args, env=env)
                else:
                    encoder_inst.process(cmd_args)
                    enc_return_code = 0
                timeslot_dump.set_end_time()
                print(f"Encoder time with all loadings, calculations and etc. is {timeslot_dump.to_seconds()} seconds")

                if enc_return_code != 0:
                    enc_finished_successfully = False
                    print(f'Encoder was terminated with error, return code {enc_return_code}')
                fp.close()
                torch.cuda.empty_cache()

            if run_dec:
                compute_stage = "decoder"
                fp.open(dec_log_path)
                ori_file = os.path.abspath(os.path.join(
                    in_dir, input_img_bn)) if not (in_dir is None) and calc_decoder_metrics else None
                cmd, cmd_args = set_cmd_for_dec(python_path, bit_path, dec_rec_path, gpu_id, **kwargs)
                #cmd_args += additional_cmd_args
                cmd_args.append("--calc_ptflops")
                if calc_decoder_metrics:
                    cmd_args.append("--calc_metrics")
                    cmd_args.append("--ori_file")
                    cmd_args.append(ori_file)            
                    cmd_args += metrics_args
                cmd_args.append('--models_dir_name')
                cmd_args.append(models_dir_name)
                if skip_loading_error:
                    cmd_args.append('--skip_loading_error')

                # cmd = cmd + ce_args
                device = 'GPU {}'.format(gpu_id) if (gpu_id >= 0) else 'CPU'
                print('Start decoder on {} : {}\n'.format(device, ' '.join(cmd + cmd_args)))
                if decoder_inst is None:
                    dec_return_code = start_and_log_str(cmd + cmd_args, env=env)
                else:
                    dec_return_code = decoder_inst.process(cmd_args)

                if dec_return_code != 0:
                    dec_finished_successfully = False
                    print('Decoder was terminated with error, return code {}'.format(
                        dec_return_code))
                fp.close()
                torch.cuda.empty_cache()

            if (coding_type == 'enc_dec' and enc_finished_successfully
                    and dec_finished_successfully) or force_encdec_match:
                compute_stage = "comparing"
                # Start tool for comparing MD5s files
                comp_log_path = os.path.join(out_dir, 'log', 'compare')
                os.makedirs(comp_log_path, exist_ok=True)
                path = os.path.join(comp_log_path, f'{input_img_fn}_{bpp_val:03d}.txt')
                fp.open(path)
                if compare_inst is None:
                    if set_cmd_for_cmp is not None:
                        cmd = set_cmd_for_cmp(python_path, out_dir, input_img_fn, bpp_val)
                        print('Start comparing: {}\n'.format(' '.join(cmd)))
                        start_and_log_str(cmd, None)
                else:
                    compare_inst.process(enc_log_path, dec_log_path)
                fp.close()
            torch.cuda.empty_cache()
        except BaseException as e:
            with open(os.path.join(out_dir, "failed.logs"), 'a') as f:
                f.write(f"Failed for {input_img_bn} (bpp on {bpp_val * 0.01}) on {compute_stage} stage. Exception args: {e.args}\n")
            if sys.gettrace() is not None:
                raise e
            else:
                print(f"Error raised. Arguments: {e.args}")
        finally:
            fp.finish()


    @staticmethod
    def compute_with_threads(gpu_list, configs):
        num_threads = len(gpu_list)

        if num_threads == 1:
            for config in configs:
                CodecEval.compute(config)
        else:
            with Pool(max_workers=num_threads) as fp:
                fp.map(CodecEval.compute, configs)


    def create_output_dirs(self, coding_type, out_dir, overwrite):
        dirs = []
        if coding_type == 'enc' or coding_type == 'enc_dec':
            dirs.append(os.path.join('log', 'enc'))
            dirs.append('rec')
            dirs.append('bit')
            dirs.append('ori')

        if coding_type == 'dec' or coding_type == 'enc_dec':
            dirs.append(os.path.join('log', 'dec'))
            dirs.append('rec_dec')

        if coding_type == 'enc_dec':
            dirs.append(os.path.join('log', 'compare'))

        create_dir_structure(out_dir, dirs, overwrite=overwrite, ignore_existance=True)

        copy_eval_scripts(out_dir, self.task_name)

        o_rec_p = 'rec' if (coding_type == 'enc') else 'rec_dec'
        return dirs, o_rec_p

    @staticmethod
    def remove_result_files(out_dir, store_bit, store_rec):
        if store_bit == 0:
            shutil.rmtree(os.path.join(out_dir, 'bit'))

        if store_rec == 0:
            shutil.rmtree(os.path.join(out_dir, 'ori'))
            shutil.rmtree(os.path.join(out_dir, 'rec'), ignore_errors=True)
            shutil.rmtree(os.path.join(out_dir, 'rec_dec'), ignore_errors=True)

    @staticmethod
    def print_device_info(only_cpu, gpu_list):
        if only_cpu:
            print('Start on {} CPUs'.format(len(gpu_list)))
        else:
            print('Start on {} GPUs'.format(', '.join(str(x) for x in gpu_list)))

    # ######################################################################################################################
    #  Main methods
    # ######################################################################################################################
    def codec_stream(self,
                     ce_args: Dict,
                     gpu_list: List,
                     params: Dict,
                     set_cmd_for_enc,
                     set_cmd_for_dec,
                     set_cmd_for_cmp,
                     **kwargs):
        """
        compute image compression pipeline on all images described in in_dir

        results will be written to out_dir (3 subdirectories):
            - ori: contains a copy of original frames
            - bit: contains the compressed bitstreams
            - rec: contains the reconstructed frames
            - rec_enc: contains the reconstructed frames of encoder if coding_type is equal to 2
        """
        
        coding_type = kwargs.get('coding_type', 'enc_dec')
        out_dir = kwargs.get('out_dir', 'results/test')
        overwrite = kwargs.get('overwrite', False)
        only_cpu = kwargs.get('only_cpu', False)
        in_dir = kwargs.get('in_dir', 'data/test')
        calc_decoder_metrics = kwargs.get('calc_decoder_metrics', 0)
        calc_encoder_metrics = kwargs.get('calc_encoder_metrics', 0)
        store_bit = kwargs.get('store_bit', 0)
        store_rec = kwargs.get('store_rec', 0)
        only_base_config = kwargs.get('only_base_config', False)
        use_yuv = kwargs.get("use_yuv", False)
        
        self.create_output_dirs(coding_type, out_dir, overwrite)

        self.print_device_info(only_cpu, gpu_list)

        dataset = ImageDataset(in_dir, lst=kwargs.get("imgs", None), ext="yuv" if use_yuv else "png")

        cfg_output = os.path.join(out_dir, 'cfg.json')
        with open(cfg_output, 'w') as fp:
            json.dump(params, fp, indent='\t')
            
        failed_logs = os.path.join(out_dir, "failed.logs")
        if os.path.exists(failed_logs):
            os.remove(failed_logs)

        configs = []
        for img_path in dataset:
            cfg_base = [cfg_output]
            img_bn = os.path.basename(img_path)
            img_fn, _ = os.path.splitext(img_bn)
            # Add configuration from per-image directory
            if not only_base_config:
                path = os.path.join(get_cfg_def_dir(), 'per-image', '{}.json'.format(img_fn))
                if os.path.exists(path):
                    cfg_base.append(path)
            
            for bpp in params.get('target_bpps', list()):
                cfg_cur = cfg_base.copy()
                # Add configuration from per-image and per-bpp
                if not only_base_config:
                    path = os.path.join(get_cfg_def_dir(), 'per-image-per-bpp', img_fn, f'bpp{bpp}.json')
                    if os.path.exists(path):
                        cfg_cur.append(path)
                config = {
                    'bpp_val': bpp,
                    'gpu_list': gpu_list,
                    'img_path': img_path,
                    'cfg_output': cfg_cur,
                    'set_cmd_for_enc': set_cmd_for_enc,
                    'set_cmd_for_dec': set_cmd_for_dec,
                    'set_cmd_for_cmp': set_cmd_for_cmp,
                    'ce_args': ce_args,
                    'kwargs': kwargs,
                    'metrics_args': self.metrics_args
                }
                configs.append(config)

        self.compute_with_threads(gpu_list, configs)

        # Execute evaluation scripts
        for i, val in enumerate( (calc_encoder_metrics, calc_decoder_metrics ) ):
            if val:
                execute_eval(out_dir, calc_decoder_metrics=(i==1))

        self.remove_result_files(out_dir, store_bit, store_rec)

    @classmethod
    def setup_device_param(cls, kwargs, params):
        if kwargs.get('only_cpu', False) or not torch.cuda.is_available():
            set_param_recurrent(params, 'target_device', 'cpu')
        return params

    def process(self, set_cmd_for_enc, set_cmd_for_dec, set_cmd_for_cmp, ce = None, **kwargs_proc):
        self.print_coder_info()

        self.init_ce(ce)
        self.init_metric_proc()        

        self.ce.get_params_list_recursively(self.parser_decorator)

        kwargs, ce_args = self.base_parser.parse_known_args()
        kwargs, ce_params = self.update_kwargs_params(kwargs,
            params_preprocess=lambda x: self.setup_device_param(vars(kwargs), x))
        del kwargs['cfg']
        rebuild_ae_cache =  kwargs.get('rebuild_ae_cache', False)

        tmp_ac = self.ce.EC(rebuild_ae_cache=rebuild_ae_cache)

        gpu_ena_list = config_gpu_list(kwargs)

        print(ce_params)

        timeslot = Timeslot()
        timeslot.set_bgn_time()
        
        if len(gpu_ena_list) > 1:
            # Remove enc/dec/compare instances if we use several GPUs
            for k in ['encoder_inst', 'decoder_inst', 'compare_inst']:
                if k in kwargs_proc:
                    del kwargs_proc[k]

        self.codec_stream(**kwargs,
                          **kwargs_proc,
                          ce_args=ce_args,
                          gpu_list=gpu_ena_list,
                          params=ce_params,
                          set_cmd_for_enc=set_cmd_for_enc,
                          set_cmd_for_dec=set_cmd_for_dec,
                          set_cmd_for_cmp=set_cmd_for_cmp)

        timeslot.set_end_time()
        timeslot.print_all_times()
