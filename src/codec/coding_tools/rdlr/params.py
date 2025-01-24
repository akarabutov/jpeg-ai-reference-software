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

from ..interfaces import ParamsBase


class RdlrParamsBPP(ParamsBase):
    def __init__(self, *args, **kwargs):
        super(RdlrParamsBPP, self).__init__(*args, **kwargs)
        add_arg = self.add_single_param

        add_arg('target_bpps',
                nargs='+',
                type=float,
                default=[0.002, 0.007, 0.015, 0.05],
                help='Betas that is used for optimization')


class RdlrCompParams(ParamsBase):
    def __init__(self, *args, **kwargs):
        super(RdlrCompParams, self).__init__(*args, **kwargs)
        add_arg = self.add_single_param

        add_arg('numIteY', type=int, default=40, help='Number of RDLR iterations for Y iteration.')

        add_arg('numIteZ', type=int, default=14, help='Number of RDLR iterations for Z')

        add_arg('lossTypeBDcurveSlope_ratioPSNR_MSSSIM',
                type=float,
                default=0.8,
                help='0: only PSNR. 1: only MSSSIM')

        add_arg('LearningRateZ',
                type=float,
                default=0.05,
                help='Learning rate for z_hat optimization in RDLR.')

        add_arg('LearningRateZDecay',
                type=float,
                default=0.9,
                help='Decay factor for learning rate in z_hat optimization.')

        add_arg('LearningRateY',
                type=float,
                default=0.03,
                help='Learning rate for y_hat optimization in RDLR.')

        add_arg('LearningRateYAutomaticPerResolution',
                type=int,
                default=1,
                help='Learning rate for y_hat optimization in RDLR is chosen automatically.'
                'This overwrites RDLRLearningRateY')

        add_arg('LearningRateYDecay',
                type=float,
                default=0.7,
                help='Decay factor for learning rate in y_hat optimization.')


class RdlrParamsCommon(ParamsBase):
    def __init__(self, *args, **kwargs):
        super(RdlrParamsCommon, self).__init__(*args, **kwargs)
        add_arg = self.add_single_param

        add_arg('numIteZLuma',
                type=int,
                default=14,
                help='Number of RDLR iterations for Z, luma component')

        add_arg('numIteZChroma',
                type=int,
                default=14,
                help='Number of RDLR iterations for Z, chroma component')

        add_arg('numSamples',
                type=int,
                default=30000001,
                help=' number of samples threshold for RDLR')

        add_arg('numSamplesPerLumaTile',
                type=int,
                default=1048576,
                help='number of samples threshold for RDLR tiles. '
                'When image has more samples than this tiles are used.')

        add_arg('numSamplesPerChromaTile',
                type=int,
                default=500000,
                help='number of samples threshold for RDLR tiles. '
                'When image has more samples than this tiles are used.')

        add_arg('numSamplesTileOverlapLuma',
                type=int,
                default=64,
                help='number of samples used for overlaping tiles in RDLR. '
                'Should be multiple of alignment size determined by number of downsampling layers')

        add_arg('numSamplesTileOverlapChroma',
                type=int,
                default=64,
                help='number of samples used for overlaping tiles in RDLR.'
                'Should be multiple of alignment size determined by number of downsampling layers')

        add_arg('lossTypeBDcurveSlope_data_source',
                type=str,
                default='single_seq_bpp_slopes',
                choices=['dataimg_all', 'single_seq_bpp_slopes'],
                help='select how to get cuvre slope data. dataimg_all is for develop. '
                'single_seq_bpp_slopes requires per-seq-per-bpp cfg files')

        add_arg('lossTypeBDcurveSlope_df', type=str, help='path to dataimg with slope data')

        add_arg('lossTypeBDcurveSlope_psnrY_slope',
                type=float,
                help='path to dataimg with slope data')

        add_arg('lossTypeBDcurveSlope_psnrU_slope',
                type=float,
                help='path to dataimg with slope data')

        add_arg('lossTypeBDcurveSlope_psnrV_slope',
                type=float,
                help='path to dataimg with slope data')

        add_arg('lossTypeBDcurveSlope_msssimY_slope',
                type=float,
                help='path to dataimg with slope data')

        add_arg('lossTypeBDcurveSlope_ratioPSNR_MSSSIM',
                type=float,
                default=0.8,
                help='0: only PSNR. 1: only MSSSIM')

        add_arg('numIteYLuma',
                type=int,
                default=40,
                help='Number of RDLR iterations for Y iteration, luma component.')

        add_arg('numIteYChroma',
                type=int,
                default=40,
                help='Number of RDLR iterations for Y iteration, chroma component.')

        add_arg('LearningRateZ',
                type=float,
                default=0.05,
                help='Learning rate for z_hat optimization in RDLR.')

        add_arg('LearningRateZDecay',
                type=float,
                default=0.9,
                help='Decay factor for learning rate in z_hat optimization.')

        add_arg('LearningRateY',
                type=float,
                default=0.03,
                help='Learning rate for y_hat optimization in RDLR.')

        add_arg('LearningRateYAutomaticPerResolution',
                type=int,
                default=1,
                help='Learning rate for y_hat optimization in RDLR is chosen automatically.'
                'This overwrites RDLRLearningRateY')

        add_arg('LearningRateYDecay',
                type=float,
                default=0.7,
                help='Decay factor for learning rate in y_hat optimization.')
