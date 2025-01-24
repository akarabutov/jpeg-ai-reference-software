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

from ...interfaces import ParamsBase
from src.codec.components import (
        DecoderFactory,
        EncoderFactory,
        HyperDecoderFactory,
        HyperEncoderFactory,
        HyperScaleDecoderFactory
    )       

class CcsCommonParams(ParamsBase):
    """Common parameters for all layers of CCS model
    """
    def __init__(self, *args, **kwargs):
        super(CcsCommonParams, self).__init__(*args, **kwargs)

        add_arg = self.add_single_param
        add_arg('BDL_range',
                type=int,
                default=[],
                nargs='+',
                help='Range of beta where the model should operate')
        add_arg('base_model_beta',
                type=float,
                default=0.002,
                help='base_model_beta of the module')
        
        add_arg('numHorRegions', type=int, default=1, help='')
        add_arg('numVerRegions', type=int, default=1, help='')
        add_arg('region_partitioning_flag', type=int, default=0, help='')
        add_arg('hyper_decoder_overlap_in_latent_samples', type=int, default=2, help='')
        add_arg('mcm_overlap_in_latent_samples', type=int, default=8, help='')
        add_arg('region_residual_in_its_own_substream_flag', type=int, default=0,
                help='Offset-based bitstream structure if 0 (all dependent regions), '
                     'marker-based bitstream structure if 1 (all independent regions)')
        add_arg('NumSamplesInRegion', type=int, default=-1, help='')
        add_arg('num_threads_z', type=int, default=1, help=r"Number of threads in substream of hyper information (z)")
        add_arg('BDL_clipping_range',
                type=int,
                default=[-1069,702],
                nargs='+',
                help='Clip the BDL that out of the range')
        
          
        
class CcsSepChannelParams(ParamsBase):
    """Common parameters for all layers of CCS model
    """
    def __init__(self, *args, **kwargs):
        super(CcsSepChannelParams, self).__init__(*args, **kwargs)

        add_arg = self.add_single_param
        add_arg('z_offset', type=int, default=31, help='z_offset value for all models')
        add_arg('z_range', type=int, default=63, help='z_range value for all models')
        add_arg('num_threads_r', type=int, default=1, help=r"Number of threads in substream of residuals")
        


class CcsSharedModelsParams(ParamsBase):
    """Parameters of modules shared between encoder and decoder
    """
    def __init__(self, *args, **kwargs):
        super(CcsSharedModelsParams, self).__init__(*args, **kwargs)

        add_arg = self.add_single_param
        add_arg('abs_in_hyperprior',
                type=int,
                default=1,
                choices=[0, 1],
                help='To use or not to use abs in hyperprior')
        add_arg('hyper_encoder_type',
                default=kwargs.get('hyper_encoder', 'basic'),
                choices=kwargs.get('hyper_encoder_choices', HyperEncoderFactory().keys()),
                help='Name of encoder network Hyper transform AE')
        add_arg('hyper_decoder_type',
                default=kwargs.get('hyper_decoder', 'basic'),
                choices=kwargs.get('hyper_decoder_choices', HyperDecoderFactory().keys()),
                help='Name of decoder network Hyper transform AE')
        add_arg('hyper_scale_decoder_type',
                default=kwargs.get('hyper_scale_decoder', 'hsd'),
                choices=kwargs.get(
                    'hyper_scale_decoder_choices',
                    HyperScaleDecoderFactory().keys()
                ),
                help='Name of decoder network Hyper SCALE transform AE')
        add_arg('use_context_module',
                type=int,
                default=0,
                choices=[0, 1],
                help='Use local context in y_hat for context modeling and GMM')        
        add_arg('num_chs',
                default=kwargs.get('num_chs', None),
                type=int,
                help='Number of channels in the substream')
        add_arg('num_decode_chs',
                default=kwargs.get('num_decode_chs', None),
                type=int,
                help='Number of decoded channels of latent space for progressive decoding. Should be multiple of 16')        
        add_arg('sigma_quant_level', type=int, default=32, help='quantization levels of sigma')
        add_arg('sigma_quant_max', type=float, default=54.82, help='quantization max of sigma')
        add_arg('sigma_quant_min', type=float, default=0.11, help='quantization min of sigma' )
        add_arg('sigma_bound_offset', type=float, default=0.5, help='boundary offset of sigma')


class CcsSingleModelParams(ParamsBase):
    """Parameters of Luma and Chroma models
    """
    def __init__(self, *args, **kwargs):
        super(CcsSingleModelParams, self).__init__()

        add_arg = self.add_single_param

        add_arg('chs_ls',
                default=kwargs.get('chs_ls', 128),
                type=int,
                nargs="+",
                help='Number of channels of latent space')
