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
    
    
import sys
##
from argparse import ArgumentParser

from src.codec.entropy_coding import ECDump
from src.codec.coders.coder import CodecCoder
from typing import List
from src.codec.common import ArgParserDecorator as ParserDecorator




def def_decoder_base_parser(task_name: str):
    this = ArgumentParser(prog=f'{task_name} bitstream probe')

    this.add_argument('INPUT_BIN', type=str, help='Path to the input bin file')
    this.add_argument('--json_output', type=str, default=None, help='Path to the output json file with dump of the bitstream structure')
    this.add_argument('--silent', action="store_true", default=False, help="Don't print result in output")

    return this

def def_eval_parser_decorator(base_parser: ArgumentParser):
    this = ParserDecorator(base_parser)

    return this


# ######################################################################################################################
#  BitstreamProbe
# ######################################################################################################################
class BitstreamProbe(CodecCoder):
    def __init__(self):
        parser = def_decoder_base_parser('Probe')
        super(BitstreamProbe, self).__init__('Probe',
                                           'Probe',
                                           parser,
                                           def_eval_parser_decorator(ArgumentParser()),
                                           is_encoder=False)

    # ##################################################################################################################
    #  decode stream
    # ##################################################################################################################
    def process_stream(self, **kwargs):
        INPUT_BIN = kwargs.get('INPUT_BIN')
        silent = kwargs.get('silent', False)
        json_output = kwargs.get('json_output', None)
         
        dump_tool = ECDump()

        self.open_bs(INPUT_BIN, dump_tool=dump_tool)
        self.close_bs()
        
        if json_output is not None:
            dump_tool.store(json_output)
        
        if not silent:
            dump_tool.print()
        
        


def process_probe(coder: BitstreamProbe, cmd_args: List[str] = None):
    """main for decoder
    """
    if cmd_args is None:
        cmd_args = sys.argv[1:]
    cmd_args.append('-loglevel')
    cmd_args.append('critical')
    kwargs, params, _ = coder.init_common_codec(build_model=False, cmd_args=cmd_args, cmd_args_add=True, add_metrics_params=False)

    coder.process_stream(**kwargs)



if __name__ == '__main__':
    codec = BitstreamProbe()
    process_probe(codec)
