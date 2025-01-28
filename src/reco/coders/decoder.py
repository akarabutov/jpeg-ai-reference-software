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
from src.codec.common import Image, Decisions
from src.codec.common.timeslot import Timeslot

from ...codec.coders import (CodecDecoder, def_decoder_base_parser, def_decoder_parser_decorator)
from ...codec.scripts import CoderProcess


# ######################################################################################################################
#  Parameter methods
# ######################################################################################################################
def def_base_parser():
    this = def_decoder_base_parser('Reconstruction')

    return this


# ######################################################################################################################
#  RecoDecoder
# ######################################################################################################################
class RecoDecoder(CodecDecoder):
    def __init__(self, base_parser, parser_decorator):
        super(RecoDecoder, self).__init__('reco', base_parser, parser_decorator)

    # ##################################################################################################################
    #  decode stream
    # ##################################################################################################################
    def decode_stream(self, bit_fpath, rec_file:str, params: dict) -> Decisions:
        if self.ce.target_device == 'cpu':
            torch.set_num_threads(1)

        basename = os.path.basename(bit_fpath)
        _, self.img_name, self.target_bpp = self.parse_bitstream_name(
            basename
        )

        self.open_bs(bit_fpath)
        # self.init_ec_module()

        decisions = self.ce.decode(self.ec_module, with_headers=False)
        self.ce.check_complience()
        self.rec_image = self.ce.decompress(decisions)

        self.close_bs()

        return decisions


def process_decoder(coder: RecoDecoder, cmd_args: List[str] = None, loadNbuild_models=True, ce = None, cmd_args_add = False, overload_ce = True):
    """main for decoder
    """
    coder.print_coder_info()

    kwargs, params, _ = coder.init_common_codec(build_model=loadNbuild_models, cmd_args=cmd_args, ce=ce, overload_ce=overload_ce, cmd_args_add=cmd_args_add)
    #print(params)
    if kwargs.get('device') == 'gpu':
        coder.init_cuda()

    coder.setup_ptflops_custom_hooks()
    
    bit_fpath = kwargs.get('bit_fpath')
    rec_path = kwargs.get('rec_path')
    calc_ptflops = kwargs.get('calc_ptflops')

    out_dir = os.path.dirname(os.path.dirname(bit_fpath))
    img_name = os.path.splitext(os.path.basename(bit_fpath))[0]
    coder.set_collector_dir(out_dir)

    if loadNbuild_models:
        timeslot_loadmodel = Timeslot()
        timeslot_loadmodel.set_bgn_time()
        coder.ce.load_models_recursively(get_downloader(kwargs.get('models_dir_name', 'models'), critical_for_file_absence=not kwargs.get('skip_loading_error', False)))
        timeslot_loadmodel.set_end_time()

    if calc_ptflops:
        if overload_ce:
            coder.ptflops_init()
        else:
            coder.ptflops_reset()

    timeslot = Timeslot()
    timeslot.set_bgn_time()

    coder.decode_stream(bit_fpath, None, kwargs)

    if kwargs.get('device') == 'gpu':
        torch.cuda.synchronize()
    timeslot.set_end_time()
    total_seconds = timeslot.to_seconds()
    timeslot.print_gap_time()

    timeslot_hash = Timeslot()
    timeslot_hash.set_bgn_time()
    coder.print_image_hash(coder.rec_image)
    timeslot_hash.set_end_time()

    timeslot_dump = None
    calc_metrics = kwargs.get('calc_metrics', False)

    timeslot_dump = Timeslot()
    timeslot_dump.set_bgn_time()
    coder.rec_image.write_file(rec_path, bit_depth=kwargs.get('output_bit_depth'))
    timeslot_dump.set_end_time()

    if calc_metrics:
        import tempfile
        ori_file = kwargs.get('ori_file', '')
        ori_ext = os.path.splitext(ori_file)[1]
        fname_suffix = ori_ext
        if ori_ext.endswith(".yuv"):
            s = coder.rec_image.shape
            fname_suffix = f"{s[-1]}x{s[-2]}_{coder.rec_image.bit_depth}bit_YUV{coder.rec_image.format}{fname_suffix}"
        with tempfile.NamedTemporaryFile(suffix=fname_suffix) as f:
            coder.compute_metrics(f.name, ori_file, bit_fpath, output_fn=os.path.basename(rec_path))
    
    if calc_ptflops:
        rec_shape = (coder.rec_image.shape[-1], coder.rec_image.shape[-2])
        coder.ptflops_term(rec_shape, rec_path, kwargs,
                        total_seconds)

    if loadNbuild_models:
        print(f'Loading models: {timeslot_loadmodel.to_seconds()} second')
    if timeslot_dump is not None:
        print(f'Dump to file: {timeslot_dump.to_seconds()} second')
    print(f'Hash calculation: {timeslot_hash.to_seconds()} second')
    coder.save_profilers_results(img_name, None)    
    return 0

class RecoDecoderProcess(CoderProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        base_parser = def_base_parser()
        parser_decorator = def_decoder_parser_decorator(base_parser)

        self.dec_inst = RecoDecoder(base_parser, parser_decorator)        
        
    def process(self, cmd_args: List[str]) -> Decisions:
        self.ce.is_encoder = False
        ans = process_decoder(self.dec_inst, cmd_args, not self.is_model_loaded(), self.ce, cmd_args_add=not self.is_args_stored(), overload_ce=self.is_first_time())
        self.set_model_loaded(True)
        self.set_first_fime(False)
        self.set_args_stored(True)
        return ans



if __name__ == '__main__':
    reco_dec = RecoDecoderProcess(None)
    reco_dec.process(None)
