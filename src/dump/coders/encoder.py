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

from src.codec.common import Decisions
from typing import List

from .common import CommonDump, def_dump_arguments
from ...reco.coders import RecoEncoder, def_reco_base_parser, reco_encoder_main
from ...codec.coders import def_encoder_parser_decorator
from ...codec.scripts import CoderProcess

##


# ######################################################################################################################
#  Parameter methods
# ######################################################################################################################

def def_base_parser():
    this = def_reco_base_parser('Dump')

    return def_dump_arguments(this)


# ######################################################################################################################
#  RecoEncoder
# ######################################################################################################################
class DumpEncoder(RecoEncoder):
    def __init__(self, base_parser, parser_decorator, name='dump'):
        super(DumpEncoder, self).__init__(base_parser, parser_decorator, name=name)
        
    # ##################################################################################################################
    #  encode stream
    # ##################################################################################################################
    
    def encode_stream(self, params):
        decisions = super().encode_stream(params)
        
        
        latents_dict = CommonDump.get_latents_dict(params['latents_list'])
        out_dir = os.path.dirname(os.path.dirname(params['bin_path']))
        seq_name = os.path.basename(params['input_path'])
        bpp_int = int(self.ce.get_target_bpp() * 100)       
        model_name = self.ce.model.tool

        filename = '{seq_name}_{bpp_int}'.format(
            codec_name=self.codec_name,
            seq_name=seq_name,
            bpp_int=bpp_int)
        
        CommonDump.store_tensor_recurrently(out_dir, latents_dict, filename, decisions[model_name], params['output_format'], params['pgx_float_scale_factor'])


class DumpEncoderProcess(CoderProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        base_parser = def_base_parser()
        self.enc_inst = DumpEncoder(base_parser, def_encoder_parser_decorator(base_parser))
        
    def process(self, cmd_args: List[str]) -> Decisions:
        self.ce.is_encoder = True
        ans = reco_encoder_main(self.enc_inst, cmd_args, not self.is_model_loaded(), self.ce, self.is_first_time(), overload_ce=self.is_first_time())
        self.set_model_loaded(True)
        self.set_first_fime(False)
        return ans

if __name__ == '__main__':
    
    reco_enc = DumpEncoderProcess(None)
    reco_enc.process(None)
