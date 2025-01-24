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
import sys
import argparse
from typing import Tuple, Dict, List
from src.codec.scripts import CodecEval, def_eval_parser_decorator

from src.codec import (CTC_get_default_fn, get_codec_name, get_codec_version, get_downloader)


def def_eval_base_parser():
    prog = 'Codec: name={}, version={}'.format(get_codec_name(), get_codec_version())
    this = argparse.ArgumentParser(prog, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    this.add_argument('--cfg',
                      default=CTC_get_default_fn(),
                      nargs='+',
                      help='path to config file(s)')
    this.add_argument('--out_dir', type=str, help='output directory', default='results/onnx')
    this.add_argument('--skip_models_check',
                      default=False,
                      action='store_true',
                      help='Skip models check and downloading')
    
    
    return this

class ModelsExporter(CodecEval):
    def __init__(self):
        parser = def_eval_base_parser()
        super(ModelsExporter, self).__init__('onnx', parser, def_eval_parser_decorator(parser))
        
    def codec_stream(self, ce_args: Dict, gpu_list: List, params: Dict, set_cmd_for_enc, set_cmd_for_dec, set_cmd_for_cmp, **kwargs):
        self.ce.eval()
        self.ce.build_models_recursively()
        self.ce.load_models_recursively(get_downloader())
        self.ce.export_models_recursively(kwargs['out_dir'], opset_version=11)        
        # Store EC
        for ec_name in self.ce.EC.factory.keys():
            a = self.ce.EC.factory.create_instance(name=ec_name)
            a.store_parameters(os.path.join(kwargs['out_dir'], ec_name))
        os.system(f"{sys.executable} -m scripts.get_z_distributions_idxs")

if __name__ == "__main__":
    oe = ModelsExporter()
    oe.process(None, None, None)