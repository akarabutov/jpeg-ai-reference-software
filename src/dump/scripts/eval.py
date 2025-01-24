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

import argparse

from ...codec.scripts import (CodecEval, def_eval_base_parser, def_eval_parser_decorator)
from ..coders import def_dump_arguments, DumpEncoderProcess, DumpDecoderProcess

##


def def_base_parser():
    this = def_eval_base_parser()

    return def_dump_arguments(this)


def set_cmd_for_enc(python_path, input_path, output_path, rec_path, bpp_val, **kwargs):
    cmd = [
        python_path, '-m', 'src.dump.coders.encoder'
    ]
    args = [input_path, output_path, 
            '--output_format', kwargs.get('output_format'),
            '--pgx_float_scale_factor', str(kwargs.get('pgx_float_scale_factor'))
            ]
    if rec_path is not None:
        args += ['-r', rec_path]
    latents_list = kwargs.get('latents_list', None)
    if latents_list is not None:
        args += ['--latents_list'] + latents_list
    args += [
        '-target_bpps', str(bpp_val), '--cfg'
    ]
    return cmd, args                        

def set_cmd_for_dec(python_path, bit_path, out_rec_path, gpu_id, **kwargs):
    cmd = [
        python_path, '-m', 'src.dump.coders.decoder'
    ]
    args = [bit_path,  out_rec_path, '--device', 'gpu' if gpu_id >= 0 else 'cpu',
            '--output_format', kwargs.get('output_format'),
            '--pgx_float_scale_factor', str(kwargs.get('pgx_float_scale_factor')),
            '--latents_list'
            ] + kwargs.get('latents_list')
    return cmd, args


if __name__ == '__main__':
    """main method
    """
    base_parser = def_base_parser()
    parser_decorator = def_eval_parser_decorator(argparse.ArgumentParser())

    coder = CodecEval('reco', base_parser, parser_decorator)
    dump_enc = DumpEncoderProcess(coder)
    dump_dec = DumpDecoderProcess(coder)
    coder.process(set_cmd_for_enc=set_cmd_for_enc,
                  set_cmd_for_dec=set_cmd_for_dec,
                  set_cmd_for_cmp=None,
                  encoder_inst=dump_enc,
                  decoder_inst=dump_dec)
