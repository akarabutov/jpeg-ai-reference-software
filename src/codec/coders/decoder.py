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
import re
##
from argparse import ArgumentParser

from src.codec import CTC_get_default_fn
from src.codec.common import ArgParserDecorator as ParserDecorator
from src.codec.common import Image, Decisions
from src.codec.metrics import MetricsProcessor

from .. import get_codec_name, get_codec_version, get_correct_rec_name
from .coder import CodecCoder, def_base_parser


def def_decoder_base_parser(task_name: str):
    this = def_base_parser(f'{task_name} Decoder', has_cfg=False)

    this.add_argument('bit_fpath', default='', help='Path of bitstream file.')
    this.add_argument('rec_path',
                      type=str,
                      default=None,
                      help='Path to output reconstructed file (in PNG)')
    
    this.add_argument('--ori_file', default=None, type=str, help='Path to original file')
    this.add_argument('--device', default='gpu', choices=['cpu', 'gpu'], help='Device')
    
    this.add_argument('--calc_ptflops',
                      default=False,
                      action="store_true",
                      help='Calculate kmac/px')    

    this.add_argument(
        "--use_yuv", type=int, default=0, help=r"Use YUV input instead of png"
    )

    this.add_argument(
        "--output_bit_depth",
        type=int,
        default=None,
        help=r"Force to use this bit-depth in the reconstructed image",
    )
    return this


def def_decoder_parser_decorator(base_parser=None):
    this = ParserDecorator(base_parser)

    return this


# ######################################################################################################################
#  CodecDecoder
# ######################################################################################################################
class CodecDecoder(CodecCoder):
    def __init__(self, task_name: str, base_parser, parser_decorator):
        super(CodecDecoder, self).__init__('Decoder',
                                           task_name,
                                           base_parser,
                                           parser_decorator,
                                           is_encoder=False)


    # ##################################################################################################################
    #  decode stream
    # ##################################################################################################################
    def decode_stream(self, bit_fpath: str, rec_file: str, params: dict) -> Decisions:
        raise NotImplementedError

    # check method
    def check_codec(self, codec_name):
        if codec_name != self.codec_name:
            msg = 'Bitstream was encoded by codec/enc={}, but decoded by codec/dec={}'
            msg = msg.format(codec_name, self.codec_name)
            raise RuntimeError(msg)


    # ##################################################################################################################
    #  service methods
    # ##################################################################################################################
    def parse_bitstream_name(self, bit_name, use_yuv=False):
        if use_yuv:
            pattern = r"(?P<codec_name>.+)_(?P<seq_num>.+)_(?P<target_bpp>\d+).bits"
        else:
            pattern = r"(?P<codec_name>.+)_(?P<seq_num>\d+)_TE_(?P<target_bpp>\d+).bits"

        results = re.search(pattern, bit_name)

        codec_name, seq_name, target_bpp = None, None, None
        if results is not None:
            codec_name = results.group("codec_name")
            seq_name = "{}_TE".format(results.group("seq_num"))
            target_bpp = int(results.group("target_bpp"))
        return codec_name, seq_name, target_bpp

    def set_collector_dir(self, root_dir):
        profilers = self.ce.get_profilers()
        base_dir = os.path.join(root_dir, 'log', 'dec')
        profilers.set_directory(base_dir)
