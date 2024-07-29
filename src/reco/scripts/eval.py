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
##
from argparse import ArgumentParser
from typing import List

from ...codec.scripts import (CodecEval, def_eval_base_parser, def_eval_parser_decorator, CompareProcess)
from ..coders import RecoEncoderProcess, RecoDecoderProcess



def set_cmd_for_enc(python_path, input_path, output_path, rec_path, bpp_val, **kwargs):
    cmd = [
        python_path, '-m', 'src.reco.coders.encoder'
    ]
    args = [input_path, output_path]
    if rec_path is not None:
        args += ['-r', rec_path]
    args += [
        '-target_bpps', str(bpp_val), '--cfg'
    ]
    # args += ["-model.Qualitymap.ROI_map_indir", f"{input_path}_mask.png"] # quality map has teh same location as the input images
    return cmd, args


def set_cmd_for_dec(python_path, bit_path, out_rec_path, gpu_id, **kwargs):
    cmd = [
        python_path, '-m', 'src.reco.coders.decoder'
    ]
    args = [bit_path,  out_rec_path, '--device', 'gpu' if gpu_id >= 0 else 'cpu']
    return cmd, args


def set_cmd_for_cmp(python_path, out_dir, seq, bpp_val, **kwargs):
    cmd = [
        python_path, '-m', 'src.reco.scripts.compare_md5s', '--i1',
        os.path.join(out_dir, 'log', 'enc', '{}_{:03d}.txt'.format(seq, bpp_val)), '--i2',
        os.path.join(out_dir, 'log', 'dec', '{}_{:03d}.txt'.format(seq, bpp_val))
    ]
    return cmd

class CompareMD5(CompareProcess):
    def __init__(self):
        super(CompareMD5, self).__init__("src.reco.scripts.compare_md5s")
        
    def process(self, log_enc: str, log_dec: str) -> None:
        args = ['--i1', log_enc, '--i2', log_dec]
        self.mod.compare(args)

if __name__ == '__main__':
    """main method
    """
    base_parser = def_eval_base_parser()
    parser_decorator = def_eval_parser_decorator(base_parser)

    coder = CodecEval('reco', base_parser, parser_decorator)
    reco_enc = RecoEncoderProcess(coder)
    reco_dec = RecoDecoderProcess(coder)
    reco_cmd = CompareMD5()
    coder.process(set_cmd_for_enc=set_cmd_for_enc,
                  set_cmd_for_dec=set_cmd_for_dec,
                  set_cmd_for_cmp=set_cmd_for_cmp,
                  encoder_inst=reco_enc,
                  decoder_inst=reco_dec,
                  compare_inst=reco_cmd)
