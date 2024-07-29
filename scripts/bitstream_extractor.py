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
from src.codec.bitstream_structure.layouts_def import SubstreamLayouts 
from src.codec.coders.coder import CodecCoder
from typing import List
from src.codec.common import ArgParserDecorator as ParserDecorator




def def_decoder_base_parser(task_name: str):
    this = ArgumentParser(prog=f'{task_name} bitstream extractor')

    this.add_argument('INPUT_BIN', type=str, help='Path to the input bin file')
    this.add_argument('OUTPUT_BIN', type=str, help='Path to the output bin file')
    this.add_argument('--remove_resi_substreams', type=int, nargs="+", default=None, help="List of residuals' substreams to be removed")
    this.add_argument('--remove_ton', default=False, action="store_true", help="Remove tools header")

    return this

def def_eval_parser_decorator(base_parser: ArgumentParser):
    this = ParserDecorator(base_parser)

    return this


# ######################################################################################################################
#  CodecExtractor
# ######################################################################################################################
class CodecExtractor(CodecCoder):
    def __init__(self):
        parser = def_decoder_base_parser('Extractor')
        super(CodecExtractor, self).__init__('Extractor',
                                           'Extractor',
                                           parser,
                                           def_eval_parser_decorator(ArgumentParser()),
                                           is_encoder=False)

    def remove_resi_substream(self, marker_id: int, region_id: int) -> None:
        if marker_id in self.bs.ae_mems_dict.keys():
            assert len(self.bs.ae_mems_dict[marker_id]) > region_id
            self.bs.ae_mems_dict[marker_id][region_id] = None
            
    def remove_ton(self) -> None:
        if SubstreamLayouts.MARKER_TON in self.bs.ae_mems_dict.keys():
            self.bs.ae_mems_dict[SubstreamLayouts.MARKER_TON][0] = None        

    # ##################################################################################################################
    #  decode stream
    # ##################################################################################################################
    def process_stream(self, **kwargs):
        INPUT_BIN = kwargs.get('INPUT_BIN')
        OUTPUT_BIN = kwargs.get('OUTPUT_BIN')
        remove_resi_substreams = kwargs.get('remove_resi_substreams', False)
        remove_ton = kwargs.get('remove_ton', False)

        self.open_bs(INPUT_BIN)
        self.output_fn = OUTPUT_BIN
        # Clean-up resi substreams
        if remove_resi_substreams is not None:
            for id in remove_resi_substreams:
                self.remove_resi_substream(SubstreamLayouts.MARKER_SORP, id)
                self.remove_resi_substream(SubstreamLayouts.MARKER_SORS, id)
                
        if remove_ton:
            self.remove_ton()
        
        for aemem_in_substream in self.bs.ae_mems_dict.values():
            for aemem in aemem_in_substream:
                if aemem is not None:
                    aemem.mem_is_ready = True
        self.close_bs()
        self.bs.decode_term()


def process_extraction(coder: CodecExtractor, cmd_args: List[str] = None):
    """main for decoder
    """
    if cmd_args is None:
        cmd_args = sys.argv[1:]
    cmd_args.append('-loglevel')
    cmd_args.append('critical')    
    kwargs, params, _ = coder.init_common_codec(build_model=False, cmd_args=cmd_args, cmd_args_add=True, add_metrics_params=False)

    coder.process_stream(**kwargs)



if __name__ == '__main__':
    codec = CodecExtractor()
    process_extraction(codec)
