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
import torch
from typing import List

from src.codec import get_downloader
from src.codec.common import Image
from src.codec.coders import CodecEncoder
from src.codec.common.timeslot import Timeslot

from ...codec.coders import (def_encoder_base_parser, def_encoder_parser_decorator)
from ...codec.scripts import CoderProcess
from src.codec.common import Decisions



# ######################################################################################################################
#  Parameter methods
# ######################################################################################################################
def def_base_parser(name='Reconstruction'):
    this = def_encoder_base_parser(name)

    return this


# ######################################################################################################################
#  RecoEncoder
# ######################################################################################################################
class RecoEncoder(CodecEncoder):
    def __init__(self, base_parser, parser_decorator, name='reco'):
        super(RecoEncoder, self).__init__(name, base_parser, parser_decorator)

    # ##################################################################################################################
    #  encode stream
    # ##################################################################################################################
    def encode_stream(self, params):
        raw_image = Image.read_file(params['input_path'])
        
        #io_stream = open(params['bin_path'], 'wb')
        #self.init_ec_lib(io_stream)
        # self.create_bs(params['bin_path'])
        # self.init_ec_module()

        #self.ec_lib.encode_init()
        if self.ce.target_device == 'cpu':
            torch.set_num_threads(1)

        self.rec_image, decisions = self.ce.compress(raw_image)
        # num_residual_streams = num_Y_tiles + num_UV_tiles
        self.create_bs(params['bin_path'])
        self.init_ec_module()

        self.ce.encode(self.ec_module, decisions)

        self.close_bs()

        return decisions
  
        

# ######################################################################################################################
#  main methods
# ######################################################################################################################
def process_encoder(coder: RecoEncoder, cmd_args: List[str] = None, loadNbuild_models=True, ce = None, cmd_args_add = False, overload_ce = True):
    """main for encoder
    """
    coder.print_coder_info()

    kwargs, params, _ = coder.init_common_codec(build_model=loadNbuild_models, cmd_args=cmd_args, ce=ce, overload_ce=overload_ce, cmd_args_add=cmd_args_add)
    profiler_path = kwargs.get('profiler_path', None)
    #print(params)

    if loadNbuild_models:
        timeslot_loadmodel = Timeslot()
        timeslot_loadmodel.set_bgn_time()
        coder.load_models(get_downloader(kwargs.get('models_dir_name', 'models'), critical_for_file_absence=not kwargs.get('skip_loading_error', False)))
        timeslot_loadmodel.set_end_time()
    coder.set_target_bpp_idx(kwargs['bpp_idx'])

    out_profiler_dir = os.path.dirname(os.path.dirname(kwargs['bin_path']))
    coder.set_collector_dir(out_profiler_dir)

    timeslot = Timeslot()
    timeslot.set_bgn_time()

    decisions = coder.encode_stream(kwargs)
    coder.ce.check_complience()
    rec_path = kwargs.get('rec_path')
    rec_fpath = rec_path
    is_write_rec = rec_path is not None

    timeslot.set_end_time()
    timeslot.print_all_times()

    timeslot_hash = Timeslot()
    timeslot_hash.set_bgn_time()
    coder.print_image_hash(coder.rec_image)
    timeslot_hash.set_end_time()
    output_ext = ".png"

    calc_metrics = kwargs.get('calc_metrics', False)
    if calc_metrics:
        ori_fn = kwargs.get('input_path', None)
        if ori_fn is not None:
            output_ext = os.path.splitext(ori_fn)[1]
    
    if is_write_rec:
        timeslot_dump = Timeslot()    
        timeslot_dump.set_bgn_time()
        coder.rec_image.write_file(rec_fpath, bit_depth=kwargs.get('output_bit_depth'))
        timeslot_dump.set_end_time()
        print(f'Dump to file: {timeslot_dump.to_seconds()} second')

    if calc_metrics:
        import tempfile
        ori_fn = kwargs.get('input_path', None)
        bit_fn = kwargs.get('bin_path', None)
        timeslot_dump = Timeslot()  
        with tempfile.NamedTemporaryFile(suffix=output_ext) as f:
            timeslot_dump.set_bgn_time()
            coder.compute_metrics(f.name, ori_fn, bit_fn, output_fn=None if rec_path is None else os.path.basename(rec_path))
            timeslot_dump.set_end_time()
        print(f'Metrics calculation: {timeslot_dump.to_seconds()} second')


    if rec_path is not None:
        coder.save_profilers_results(os.path.basename(rec_path), kwargs.get('target_bpps', [1])[0])

    if loadNbuild_models:
        print(f'Loading models: {timeslot_loadmodel.to_seconds()} second')
    print(f'Hash calculation: {timeslot_hash.to_seconds()} second')
    
    return decisions

class RecoEncoderProcess(CoderProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        base_parser = def_base_parser()
        self.enc_inst = RecoEncoder(base_parser, def_encoder_parser_decorator(base_parser))
        
    def process(self, cmd_args: List[str]) -> Decisions:
        self.ce.is_encoder = True
        ans = process_encoder(self.enc_inst, cmd_args, not self.is_model_loaded(), self.ce, self.is_first_time(), overload_ce=self.is_first_time())
        self.set_model_loaded(True)
        self.set_first_fime(False)
        return ans

if __name__ == '__main__':
    reco_enc = RecoEncoderProcess(None)
    reco_enc.process(None)