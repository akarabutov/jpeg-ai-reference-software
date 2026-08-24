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

from typing import Dict, List, Tuple

import os
import torch
import commentjson

from src.codec import get_profiles_dir
from src.codec.common import Decisions, Image
from src.codec.entropy_coding import ECModule, HeaderCoder

from ..bitrate_matcher import BitrateMatcher
from ..core_models import create_core_model_instance
from ..filters import create_filters_instance
from ..interfaces import ToolEngine
from ..profiler import Profilers
from ..rdlr import RDLR
from ..resolution_changer import ResolutionChanger
from ..colour_processing import create_colour_processing_instance
from ..rdi import RDI
from ..udi import UDI
##
from .params import AicParams


class CodingEngine(ToolEngine):
    img_width = 0
    img_height = 0
    image_data_bits = 8

    max_width = (1<<16)-1+64
    max_height = (1<<16)-1+64

    max_supported_bpp = 200

    supported_image_data_bits = [8, 10, 12, 14, 16]
    __target_bpp_idx = 0

    def __init__(self, **kwargs):
        super(CodingEngine, self).__init__(has_enabled_flag=False, stream_header_part='pic_header', **kwargs)

        self.Profilers = Profilers()

        self._params_ce = AicParams()
        self.is_encoder = True
        if 'is_encoder' in kwargs:
            self.is_encoder = kwargs.get('is_encoder', False)
        self.res_changer = ResolutionChanger(is_encoder=self.is_encoder, signal_enabled_flag=False)     # TODO: remove this tool
        self.colour_processing = create_colour_processing_instance(stream_header_part='pic_header')
        self.model = create_core_model_instance(stream_header_part='pic_header')
        self.post_filters = create_filters_instance(stream_header_part='tool_header')
        self.udi = UDI()
        self.rdi = RDI()

        # RDO tools
        self.model.add_preproc_tool('bitrate_matcher', BitrateMatcher())
        self.model.add_postproc_tool('rdlr', RDLR())

        self.eval()
        
    def init_new_img(self) -> None:
        self.c_ver = 1
        self.c_hor = 1        

    def compress_model(self, img: Image):
        decisions = self.model.compress(img)
        rec_img = self.model.decompress(decisions)
        return rec_img, decisions

    def compress(self, image: Image) -> Tuple[Image, Decisions]:

        self.img_height, self.img_width = image.shape[-2:]
        self.image_data_bits = image.bit_depth

        self.image_filepath = image.get_filename()
        with torch.no_grad():
            self.get_profilers().start('image compression')
            # Generate decisions
            self.init_new_img_recursivly()
            image.to_(self.device, non_blocking=True)
            self.s_ver, self.s_hor = image.get_chroma_subsampling()
            self.c_ver = max(self.c_ver, self.s_ver) if self.c_ver_value is None else self.c_ver_value
            self.c_hor = max(self.c_hor, self.s_hor) if self.c_hor_value is None else self.c_hor_value
            assert self.c_ver >= self.s_ver
            assert self.c_hor >= self.s_hor

            img_orig = image.clone()

            # check for intra resolution change
            self.res_changer.setup([self.img_height, self.img_width])

            # Compress by model
            img_coded_resolution = self.res_changer.forward_transform(img_orig)
            
            img2compress = self.colour_processing.pre_processing(img_coded_resolution)
            img2compress.input_file = image.input_file ## add path for further operation

            rec_compressed, decisions = self.compress_model(img2compress)
            
            rec_img = self.res_changer.backward_transform(rec_compressed)
            rec_img.to_format_(image.format)

            # Execute post filters
            self.post_filters.compress(rec_img.clone(), image.clone(), decisions=decisions)
            rec_img_after_filters = self.post_filters.decompress(rec_img, decisions=decisions)

            rec_img = self.colour_processing.post_processing(rec_img_after_filters)

            self.get_profilers().finish('image compression')

            return rec_img, decisions

    def decompress(self, decisions: Decisions) -> Image:
        with torch.no_grad():
            self.get_profilers().start('image decompression')
            rec_img = self.model.decompress(decisions)        
            rec_img = self.res_changer.backward_transform(rec_img)
            rec_img.to_format_(Image.get_format_from_subsampling(self.s_ver, self.s_hor))
            
            self.logger.info('Image decompressed')
            # Execute post filters
            if self.model.is_enabled():                
                rec_img_after_filters = self.post_filters.decompress(rec_img, decisions=decisions)
            rec_img = self.colour_processing.post_processing(rec_img_after_filters)
            self.get_profilers().finish('image decompression')
            return rec_img

    def encode(self, ec: ECModule, decisions: Decisions) -> None:
        self.logger.debug('Storing data to a bitstream')
        # Store decisions to bitstream
        with self.get_profilers_ctx('image encoding'):
            self.encode_header_recursively(ec.get_header_codec())
            with self.get_profilers_ctx('model encode'):
                self.model.encode(ec, decisions)

    def decode(self, ec: ECModule, with_headers: bool = False) -> Decisions:
        if with_headers:
            self.init_new_img_recursivly()
        with self.get_profilers_ctx('image decoding'):
            if with_headers:
                self.decode_header_recursively(ec.get_header_codec())
            with torch.no_grad():
                if self.model.is_enabled():
                    decision = self.model.decode(ec)
                    self.logger.info('Image decoded')
                else:
                    raise NotImplementedError

        return decision

    def encode_header(self, ec: HeaderCoder):
        assert self.diff_display_img_width >= 0 and self.diff_display_img_width < 64
        assert self.diff_display_img_height >= 0 and self.diff_display_img_height < 64
        
        # Profile level parameters
        ec.encode(self.stream_profile_id, 
                   bits_count=4,
                   name=r'stream_profile_idc')
        ec.encode(self.decoder_profile_id,
                  bits_count=4,
                  name="decoder_profile_id")
        ec.encode(len(self.synthesis_transform_id)-1,
                  bits_count=4,
                  name="num_synthesis_transforms_minus1")
        for i,dec_id in enumerate(self.synthesis_transform_id):
            ec.encode(dec_id, 
                      bits_count=4, 
                      name=f'synthesis_transform_id[{i}]')
        ec.encode(self.level_idc,
                  bits_count=8, 
                  name="level_idc")
        # Profile level parameters end

        ec.encode(self.img_width - 64, self.max_width - 64, name='img_width_minus64')
        ec.encode(self.img_height - 64, self.max_height - 64, name='img_height_minus64')

        ec.encode(self.diff_display_img_width, bits_count=6, name='diff_display_img_width')
        ec.encode(self.diff_display_img_height, bits_count=6, name='diff_display_img_height')

        ec.encode(self.supported_image_data_bits.index(self.image_data_bits),
                  max_symbol_value=len(self.supported_image_data_bits) - 1,
                  name='bit_depth_idc')
        ec.encode(self.s_ver-1, max_symbol_value=1, name='s_ver_minus1')
        ec.encode(self.s_hor-1, max_symbol_value=1, name='s_hor_minus1')
        if self.s_ver == 1:
            ec.encode(self.c_ver-1, max_symbol_value=1, name='c_ver_minus1')
        if self.s_hor == 1:
            ec.encode(self.c_hor-1, max_symbol_value=1, name='c_hor_minus1')


    def decode_header(self, ec: HeaderCoder):
        self.stream_profile_id = int(ec.decode([1], 
                                                bits_count=4,
                                                name=r'stream_profile_idc'))
        # Profile level parameters
        self.decoder_profile_id = int(ec.decode([1],
                                                bits_count=4, 
                                                name="decoder_profile_id"))
        num_synthesis_transforms_minus1 = int(ec.decode([1],
                                                    bits_count=4, 
                                                    name="num_synthesis_transforms_minus1"))
        self.synthesis_transform_id = [0] * (num_synthesis_transforms_minus1 + 1)
        for i in range(num_synthesis_transforms_minus1 + 1):
            self.synthesis_transform_id[i] = int(ec.decode([1],
                                                        bits_count=4,
                                                        name=f'synthesis_transform_id[{i}]'))
        self.level_idc = int(ec.decode([1],
                                       bits_count=8,
                                       name="level_idc"))
        # Profile level parameters end

        self.img_width = int(ec.decode([1], self.max_width - 64, name='img_width_minus64') + 64)
        self.img_height = int(ec.decode([1], self.max_height - 64, name='img_height_minus64') + 64)

        self.diff_display_img_width = int(ec.decode([1], bits_count=6, name='diff_display_img_width'))
        self.diff_display_img_height = int(ec.decode([1], bits_count=6, name='diff_display_img_height'))
        

        image_data_bits_idx = int(
            ec.decode([1], max_symbol_value=len(self.supported_image_data_bits) - 1,
                      name='bit_depth_idc'))
        assert image_data_bits_idx in [0, 1], "Only 8-bit and 10-bit are supported."
        self.image_data_bits = self.supported_image_data_bits[image_data_bits_idx]
        self.s_ver = int(
            ec.decode([1], max_symbol_value=1, name='s_ver_minus1'))+1
        self.s_hor = int(
            ec.decode([1], max_symbol_value=1, name='s_hor_minus1'))+1
        self.c_ver = int(ec.decode([1], max_symbol_value=1, name='c_ver_minus1'))+1 if self.s_ver==1 else 2
        self.c_hor = int(ec.decode([1], max_symbol_value=1, name='c_hor_minus1'))+1 if self.s_hor==1 else 2
        self.res_changer.setup([self.img_height, self.img_width])
        
    def get_default_decoder_id(self):
        return self.synthesis_transform_id[0] if self.decoder_id is None else self.decoder_id
    
    def check_complience(self):
        # Check profile
        ## Get list of supported profiles
        with open(os.path.join(get_profiles_dir(), "profiles_list.json"), "r") as f:
            prof_list = commentjson.load(f)['profiles']
        assert self.decoder_profile_id < len(prof_list) and self.decoder_profile_id >= 0, "Incorrect value of decoder_profile_id"
        with open(os.path.join(get_profiles_dir(), f"{prof_list[self.decoder_profile_id]}.json"), "r") as f:
            profile_cfg = commentjson.load(f)
        # Sanity check
        assert profile_cfg['decoder_profile_id'] == self.decoder_profile_id, "Incorrect value of decoder_profile_id in the configuration file of the profile"
        # Check stream profile
        assert profile_cfg['stream_profile_id'] == self.stream_profile_id, "Incorrect value of stream_profile_id in the configuration file of the profile"
        # Check synthesis 
        assert self.get_default_decoder_id() in profile_cfg['synthesis_transform_id'], "The profile doesn't support the synthesis network"
        
        # Check level
        with open(os.path.join(get_profiles_dir(), "levels.json"), "r") as f:
            levels_cfg = commentjson.load(f)
        lvl0 = self.level_idc // 10
        lvl1 = self.level_idc - 10 * lvl0
        assert f"{lvl0}" in levels_cfg['level_idc0'], "Unsupported type of level"
        assert f"{lvl1}" in levels_cfg['level_idc1'], "Unsupported type of level"
        ## Check image size
        assert self.img_height * self.img_width <= levels_cfg['level_idc0'][f"{lvl0}"]["max_pic_size"], "The output image is too large"
        ## Check model_id
        assert self.model.get_tool().get_active_tool_idx() in levels_cfg['level_idc1'][f"{lvl1}"]["models"], "Unproper model is used"
        
    def build_models_recursively(self):
        super().build_models_recursively()
        self.train(self.training)

    def train(self, state=True):
        super().train(state)
        for n, m in self.named_modules():
            for pn, pp in m._parameters.items():
                if pp is not None:
                    pp.requires_grad = m.training

    # Backward hooks
    def set_target_bpp_idx(self, bpp):
        self.__target_bpp_idx = int(bpp)

    def get_target_bpp_idx(self):
        return self.__target_bpp_idx

    def get_target_bpp(self) -> float:
        idx = self.__target_bpp_idx
        ans = self.target_bpps[idx] if hasattr(
            self, 'target_bpps') and len(self.target_bpps) > 0 else idx
        return ans * 0.01

    def get_target_bpps(self) -> List[float]:
        return self.target_bpps
    
    def get_target_bpp_int(self) -> int:
        return int(self.get_target_bpp() * 100)

    def get_tool_url(self) -> str:
        return 'ce'

    def get_processed_img_shape(self):
        return self.res_changer.get_processed_img_shape()

    def get_original_img_shape(self):
        return self.res_changer.get_original_img_shape()

    def get_profilers(self):
        return self.Profilers
