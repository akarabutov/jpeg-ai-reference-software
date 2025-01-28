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

import logging
import os

from argparse import ArgumentParser
import ptflops
import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops.flops_counter import flops_to_string
from typing import List, Dict, Any

from src.codec import get_codec_name, get_codec_version, set_param_recurrent, CTC_get_default_fn, get_pipeline_desc_paths
from src.codec.coding_tools.coding_engine import CodingEngine
from src.codec.common import ArgParserDecorator as ParserDecorator
from src.codec.common import Image, Decisions
from src.codec.entropy_coding import ECFactory, ECModule, ECDump
from src.codec.bitstream_structure import BitstreamStructure
from src.codec.metrics import MetricsProcessor
from src.codec.utils import cmd_params_loading, ExtendAction
from packaging import version
from copy import deepcopy
TORCH_OLDER_THAN_1_13_1 =  version.parse(torch.__version__) < version.parse("1.13.1")



def def_base_parser(task_name: str, **kwargs):
    this = ArgumentParser(prog=f'{get_codec_name()} {task_name} [{get_codec_version()}', conflict_handler='resolve')

    if kwargs.get('has_cfg', True):
        this.add_argument('--cfg',
                        type=str,
                        nargs='+',
                        action=ExtendAction,
                        default=list(),
                        help='Path to the config file.')

    ##
    this.add_argument('-rb','--rec_bitdepth',
                      type=int,
                      default=8,
                      choices=[8, 10],
                      help='Bit depth for reconstructed file.')
    this.add_argument('--profiler_path',
                      type=str,
                      default=None,
                      help='Path to output profiler')
    
    this.add_argument('--calc_metrics',
                      default=False,
                      action="store_true",
                      help='Calculate metrics')
    
    this.add_argument('--models_dir_name',
                      type=str,
                      default="models",
                      help='Name of a directory with models')
    
    this.add_argument('--skip_loading_error', 
                      default=False,
                      action="store_true",
                      help=r"Skip error of models loading if it happends")
    
    
    return this
    

# ######################################################################################################################
# CodecCoder
# ######################################################################################################################
class CodecCoder:
    def __init__(self,
                 coder_name: str,
                 task_name: str,
                 base_parser,
                 parser_decorator: ParserDecorator,
                 is_encoder=False):
        self.codec_name = get_codec_name()
        self.codec_version = get_codec_version()
        self.coder_name = coder_name
        self.task_name = task_name
        self.is_encoder = is_encoder

        self.base_parser = base_parser
        self.parser_decorator = parser_decorator

        self.ce = None
        self.ec_module = None
        self.metric_proc = None
        self.output_fn = None

    # ##################################################################################################################
    #  Parameter methods
    # ##################################################################################################################
    def update_kwargs_params(self, kwargs, params_preprocess=None, cmd_args:List = None):
        #kwargs, unknown_params = self.base_parser.parse_known_args()
        kwargs = vars(kwargs)
        if self.is_encoder:
            cfg = CTC_get_default_fn() if len(kwargs.get('cfg')) == 0 else kwargs.get('cfg')
        else:
            cfg = get_pipeline_desc_paths()
        params = cmd_params_loading(self.parser_decorator,
                                    self.ce,
                                    cfg if self.is_encoder else get_pipeline_desc_paths(),
                                    params_preprocess=params_preprocess,
                                    cmd_args=cmd_args)

        params = self.ce.store_attrs2dict_recursively()
        return kwargs, params #, unknown_params

    # ##################################################################################################################
    #  init methods
    # ##################################################################################################################
    def init_ce(self, ce: CodingEngine):
        self.ce = ce
        profilers = self.ce.get_profilers()
        self.ce.EC = ECFactory.create_instance(profilers=profilers)

    @staticmethod
    def init_cuda():
        y = torch.ones((1, 1, 1, 1), device='cuda')
        z = F.conv2d(y, y)  # noqa: F841

    def create_bs(self, output_fn: str) -> None:
        self.bs = BitstreamStructure(self.ce.EC, coder_direction=0)
        self.output_fn = output_fn
        
    def open_bs(self, input_fn: str, dump_tool: ECDump = None) -> None:
        self.bs = BitstreamStructure(self.ce.EC, coder_direction=1, dump_tool=dump_tool) #, dec_only_pic_header=True)
        self.bs.read_substreams(input_fn)
        self.bs.parse_substreams(True)
        self.init_ec_module(dump_tool=dump_tool)
        self.ce.init_new_img_recursivly()
        self.ce.decode_header_recursively(self.ec_module.get_header_codec())
        self.bs.parse_substreams(False)
        
        self.output_fn = None
    
    def close_bs(self) -> None:
        if self.output_fn is not None:
            self.bs.fill_substreams()
            self.bs.write_substreams(self.output_fn)
            self.output_fn = None
        else:
            self.bs.decode_term()

        
    def init_ec_module(self, verbose:bool=None, dump_tool: ECDump = None):
        if verbose is None:
            verbose = self.ce.EC.verbose
        self.ec_module = ECModule(self.bs, verbose, profilers=self.ce.get_profilers(), dump_tool=dump_tool)

    # ##################################################################################################################
    #  init methods
    # ##################################################################################################################
    def init_metric_proc(self, cmd_args: List[str] = None, cmd_args_add = True):
        metric_proc = MetricsProcessor()
        ap = self.base_parser
        if cmd_args_add:
            metric_proc.add_arguments(ap)

        args = metric_proc.parse_arguments(ap, cmd_args)
        self.metric_proc = metric_proc

    def init_backend_model(self):
        pass

    def init_common_codec(self, build_model=True, cmd_args: List[str] = None, ce: CodingEngine = None, overload_ce: bool = True, cmd_args_add = False, add_metrics_params: bool = True):
        """ Init common codec:
                - init CoderEngine
                - init metrics
                - map models on devices
                - configure CoderEngine internal structure

        Returns:
            _type_: _description_
        """
        if ce is None:
            ce = CodingEngine(is_encoder=self.is_encoder)
        else:
            ce.is_encoder = self.is_encoder
        self.ce = ce

        if overload_ce:
            self.init_ce(ce)
            self.init_backend_model()

        if cmd_args_add:
            self.get_params_list()

        if add_metrics_params:
            self.init_metric_proc(cmd_args, True)

        kwargs, unknown_params = self.base_parser.parse_known_args(args=cmd_args)
        kwargs, params = self.update_kwargs_params(kwargs, 
            params_preprocess=lambda x: self.setup_device_param(vars(kwargs), x),
            cmd_args=cmd_args)

        if build_model:
            ce.eval()
            ce.build_models_recursively()
            
        return kwargs, params, unknown_params

    # ##################################################################################################################
    #  encode/decode methods
    # ##################################################################################################################
    def encode_stream(self, *arg, **kwargs) -> Decisions:
        pass

    def decode_stream(self, *arg, **kwargs) -> Decisions:
        pass

    # ##################################################################################################################
    #  Service methods
    # ##################################################################################################################


    def parse_to_kwargs(self):
        args = self.base_parser.parse_args()
        kwargs = vars(args)
        return kwargs

    def print_coder_info(self):
        print('Codec/{}: name={}, version={}'.format(self.coder_name, self.codec_name,
                                                     self.codec_version))

    def get_params_list(self):
        if self.parser_decorator is not None:
            self.ce.get_params_list_recursively(self.parser_decorator)

    @staticmethod
    def print_image_hash(rec_img: Image):
        import hashlib
        md5_lst = list()
        for c in Image.valid_comp_names:
            md5_lst.append(hashlib.md5(rec_img.get_component(c).detach().contiguous().cpu().numpy()).hexdigest())
        print(f'MD5: ({", ".join(md5_lst)})')

    @staticmethod
    def setup_total_seconds(kwargs, total_seconds):
        device = kwargs.get('device','gpu')
        dec_gpu = None if (device == 'cpu') else total_seconds
        dec_cpu = None if (device == 'gpu') else total_seconds
        return dec_gpu, dec_cpu

    @classmethod
    def setup_device_param(cls, kwargs, params):
        have_gpu = torch.cuda.is_available()
        if have_gpu:
            dev_list = os.environ.get('CUDA_VISIBLE_DEVICES', list())
            if len(dev_list) == 0:
                have_gpu = False
            else:
                corr_dev_id = False
                for x in dev_list.split(','):
                    dev_id = int(x)
                    if dev_id >= 0:
                        corr_dev_id = True
                have_gpu = corr_dev_id
                
        if kwargs.get('device', '') == 'cpu' or not have_gpu:
            set_param_recurrent(params, 'target_device', 'cpu')
            kwargs['device'] = 'cpu'
        return params

    # ##################################################################################################################
    #  ptflops methods
    # ##################################################################################################################
    def ptflops_init(self):
        self.setup_ptflops_custom_hooks()
        self.metric_proc.init_ptflops_calc(self.ce)
        
    def ptflops_reset(self):
        self.metric_proc.reset_ptflops_calc(self.ce)

    def ptflops_term(self, rec_size, rec_path, kwargs, total_seconds):
        flops_full, flops_pixel = self.metric_proc.finish_ptflops_calc(self.ce, rec_size)

        if rec_path is not None:
            t_gpu, t_cpu = self.setup_total_seconds(kwargs, total_seconds)
            if self.is_encoder:
                compl = {'encGPU': t_gpu, 'encCPU': t_cpu}
            else:
                compl = {'decGPU': t_gpu, 'decCPU': t_cpu}
            # Create directory for reconstruct images
            dir_path = os.path.dirname(rec_path)
            os.makedirs(dir_path, exist_ok=True)
            # Store complexity information
            self.metric_proc.store_complexity_info(rec_path, flops_pixel / 1000, **compl)

        msg = 'Flops: {0}, i.e {1} / pxl'
        msg = msg.format(flops_to_string(flops_full, units=None),
                         flops_to_string(flops_pixel, units='KMac'))
        print(msg)

    @staticmethod
    def get_ptflops_custom_hooks_recursively(root_module: nn.Module):
        ans = dict()
        for n, m in root_module.named_children():
            t = type(m)
            if hasattr(m, 'ptflops_custom_hook') and (t not in ans.keys()):
                ans[t] = m.ptflops_custom_hook()
            tmp = CodecCoder.get_ptflops_custom_hooks_recursively(m)
            ans.update(tmp)
        return ans

    def setup_ptflops_custom_hooks(self):
        print('Collect ptflops for complexity measurement')
        ptflops.flops_counter.CUSTOM_MODULES_MAPPING = self.get_ptflops_custom_hooks_recursively(
            self.ce)

    # collector methods
    def save_profilers_results(self, seq_name, bpp_idx):
        filename = (seq_name + '_' + str(bpp_idx)) if bpp_idx is not None else seq_name
        self.ce.get_profilers().save_results(filename)

    def set_collector_dir(self, root_dir):
        raise NotImplementedError
    
    # metric methods
    def compute_and_print_metrics(self, shape, bit_fpath, raw_fpath, rec_fpath, device = None, output_fn=None):
        bpp = self.metric_proc.compute_bpp(bit_fpath, shape)
        w, h, b, fmt = Image.extract_info(raw_fpath, default_bits=8)
        device = self.ce.device if TORCH_OLDER_THAN_1_13_1 else torch.device("cpu")
        if fmt != "sRGB" and fmt != "444":  # yuv 444 is also hanndled by process_image_files
            metrics = self.metric_proc.process_yuv_files(
                shape,
                raw_fpath,
                rec_fpath,
                device=device,
            )
        else:
            metrics = self.metric_proc.process_image_files(
                raw_fpath,
                rec_fpath,
                device=device,
            )

        basename = os.path.basename(rec_fpath) if output_fn is None else output_fn
        results = self.metric_proc.get_output_str(basename, bpp=bpp, metrics=metrics)
        print(f'=== Metrics ===\nResults: {results}\n=== Metrics ===')  
        
        
    def compute_metrics(self, rec_fn: str, ori_fn: str, bit_fn: str, output_fn: str = None) -> None:
        rec_image = self.rec_image
        rec_image.write_file(rec_fn)
        
        try:
            self.compute_and_print_metrics(rec_image.shape, bit_fn, ori_fn, rec_fn, output_fn=output_fn)
        except RuntimeError as e:
            self.compute_and_print_metrics(rec_image.shape, bit_fn, ori_fn, rec_fn, device=torch.device("cpu"), output_fn=output_fn)

    # ##################################################################################################################
    #  Working with decisions
    # ##################################################################################################################

    def extract_model_decision(self, module_url, decisions: Decisions) -> Decisions:
        model_url_list = module_url.split(".")
        cur_decision = decisions
        for sub_modul in model_url_list:
            cur_decision = cur_decision[sub_modul]
        return cur_decision
    