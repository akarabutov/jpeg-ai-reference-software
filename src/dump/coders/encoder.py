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

from argparse import ArgumentParser
from src.codec.common import Decisions
from typing import List

from ...reco.coders import RecoEncoder, def_reco_base_parser, reco_encoder_main
from ...codec.coders import def_encoder_parser_decorator
from ...codec.scripts import CoderProcess

##


# ######################################################################################################################
#  Parameter methods
# ######################################################################################################################

def def_dump_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--latents_list',
                      type=str,
                      nargs="+",
                      default=['model_y.z_hat', 'model_y.y_hat', 'model_uv.z_hat', 'model_uv.y_hat'],
                      help='List of tensors for dumping')
    
    return parser
    

def def_base_parser():
    this = def_reco_base_parser('Dump')

    return def_dump_arguments(this)


# ######################################################################################################################
#  RecoEncoder
# ######################################################################################################################
class DumpEncoder(RecoEncoder):
    def __init__(self, base_parser, parser_decorator, name='dump'):
        super(DumpEncoder, self).__init__(base_parser, parser_decorator, name=name)
        
    def get_latents_dict(self, latents_list):
        ans = dict()
        for l in latents_list:
            l_sublist = l.split(".")
            cur_ans = ans
            for item in l_sublist:
                if item not in cur_ans:
                    cur_ans[item] = dict()
                cur_ans = cur_ans[item]
        return ans

    # ##################################################################################################################
    #  encode stream
    # ##################################################################################################################
    @staticmethod
    def store_tensor_recurrently(base_path: str, latent_dict: dict, filename: str, decisions: Decisions) -> None:
        for n, v in latent_dict.items():
            path = os.path.join(base_path, n)
            if len(v) == 0:
                if isinstance(decisions[n], torch.Tensor):
                    os.makedirs(path, exist_ok=True)
                    torch.save(decisions[n], os.path.join(path, filename))
                else:
                    raise ValueError("Type of decision isn't torch tensor")
            else:
                if n in decisions:
                    DumpEncoder.store_tensor_recurrently(path, v, filename, decisions[n])
                else:
                    raise ValueError(f"Decision doesn't have {n} item")
    
    def encode_stream(self, params):
        decisions = super().encode_stream(params)
        
        
        latents_dict = self.get_latents_dict(params['latents_list'])
        out_dir = os.path.dirname(os.path.dirname(params['bin_path']))
        seq_name = os.path.basename(params['input_path'])
        bpp_int = int(self.ce.get_target_bpp() * 100)       
        model_name = self.ce.model.tool

        filename = '{seq_name}_{bpp_int}'.format(
            codec_name=self.codec_name,
            seq_name=seq_name,
            bpp_int=bpp_int)
        
        self.store_tensor_recurrently(out_dir, latents_dict, filename, decisions[model_name])


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
