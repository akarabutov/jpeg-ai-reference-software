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

from src.codec import (CTC_get_default_fn, get_codec_name, get_codec_version)
from src.codec.common import Decisions, ArgParserDecorator as ParserDecorator
from src.codec.utils import Downloader
from typing import List

from .coder import CodecCoder, def_base_parser


def def_encoder_base_parser(task_name: str):
    this = def_base_parser(f'{task_name} Encoder')

    this.add_argument('--bpp_idx', type=int, default=0, help='Index of QP to be processed.')
    ##

    this.add_argument('input_path', type=str, help='Path to input file (in PNG format)')
    this.add_argument('bin_path', type=str, help='Path to output binary file')
    
    this.add_argument('-r','--rec_path',
                      type=str,
                      default=None,
                      help='Path to output reconstructed file (in PNG)')
    
    this.add_argument('--set_target_bpp', type=int, default=None, help='Set target BPP of the final stream')

    this.add_argument(
        "--output_bit_depth",
        type=int,
        default=None,
        help=r"Force to use this bit-depth in the reconstructed image",
    )

    return this


def def_encoder_parser_decorator(parser=ArgumentParser()):
    this = ParserDecorator(parser)

    return this


# ######################################################################################################################
#  CodecEncoder
# ######################################################################################################################
class CodecEncoder(CodecCoder):
    def __init__(self, task_name: str, base_parser, parser_decorator):
        super(CodecEncoder, self).__init__('Encoder',
                                           task_name,
                                           base_parser,
                                           parser_decorator,
                                           is_encoder=True)

    # ##################################################################################################################
    #  encode stream
    # ##################################################################################################################
    def encode_stream(self, kwargs, params) -> Decisions:
        raise NotImplementedError



    # collector methods
    def set_collector_dir(self, base_dir):
        profilers = self.ce.get_profilers()
        profilers.set_directory(base_dir)

    # ##################################################################################################################
    #  service methods
    # ##################################################################################################################
    def load_models(self, downloader: Downloader) -> None:
        """load models

        Args:
            downloader (Downloader): object with information about models
        """
        self.ce.load_models_recursively(downloader)

    def set_target_bpp_idx(self, target_bpp_idx: int) -> None:
        """Set target BPP of CoderEngine

        Args:
            target_bpp_idx (int): index of target BPP in a list

        """
        self.ce.set_target_bpp_idx(target_bpp_idx)
        
    @staticmethod
    def __update_target_bpps(ans, target_bpp, params_preprocess=None):
        ans['target_bpps'] = [target_bpp]
        if params_preprocess is not None:
            ans = params_preprocess(ans)
        return ans

    def update_kwargs_params(self, kwargs, params_preprocess=None, cmd_args:List = None):
        from src.codec import get_cfg_def_dir
        set_target_bpp = kwargs.set_target_bpp
        if set_target_bpp is not None:
            cfg = CTC_get_default_fn() if len(kwargs.cfg) == 0 else kwargs.cfg
            cfg.append(os.path.join(get_cfg_def_dir(), 'BRM', 'regen_list.json'))
            kwargs.cfg = cfg
            kwargs.bpp_idx = 0            
            func=lambda x: self.__update_target_bpps(x, set_target_bpp, params_preprocess)
        else:
            func=params_preprocess
            
        return super().update_kwargs_params(kwargs=kwargs,
                                            params_preprocess=func,
                                            cmd_args=cmd_args)
            