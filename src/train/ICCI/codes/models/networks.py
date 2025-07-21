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

import torch

import models.archs.discriminator_vgg_arch as SRGAN_arch
import models.archs.SRResNet_arch as SRResNet_arch

#import models.archs.RRDBNet_arch as RRDBNet_arch
#import models.archs.EDVR_arch as EDVR_arch

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    # image restoration
    if which_model == 'ThreeStage':
        netG = SRResNet_arch.ThreeStageModel(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'])
    elif which_model == 'ThreeStageYUV':
        netG = SRResNet_arch.ThreeStageModelYUV(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'])
    elif which_model == 'ThreeStageYUV444':
        netG = SRResNet_arch.ThreeStageModelYUV444(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'])
    elif which_model == 'ThreeStageYUV_DWT':
        netG = SRResNet_arch.ThreeStageModelYUV_DWT(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'])
    elif which_model == 'YOnlyModel':
        netG = SRResNet_arch.YOnlyModel(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'])
    elif which_model == 'YOnlyModel_debug':
        netG = SRResNet_arch.YOnlyModel_debug(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'])
    elif which_model == 'ThreeStageYUVLite':
        netG = SRResNet_arch.ThreeStageYUVLite(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'])
    elif which_model == 'ThreeStageYUVLiteDWTv1':
        netG = SRResNet_arch.ThreeStageYUVLite_DWT_V1(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'])
    elif which_model == 'ThreeStageYUVLiteDWTv2':
        netG = SRResNet_arch.ThreeStageYUVLite_DWT_V2(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nbY=opt_net['nb'], nbUV=opt_net['nbUV'])
    elif which_model == 'ThreeStageYUVLiteDWTv3':
        netG = SRResNet_arch.ThreeStageYUVLite_DWT_V3(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nbY=opt_net['nb'], nbUV=opt_net['nbUV'])
    elif which_model == 'ThreeStageYUVLiteDWTv3_1551':
        netG = SRResNet_arch.ThreeStageYUVLite_DWT_V3_1551(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nbY=opt_net['nb'], nbUV=opt_net['nbUV'])
    elif which_model == 'ThreeStageYUVLiteDWTv4':
        netG = SRResNet_arch.ThreeStageYUVLite_DWT_V4(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nbY=opt_net['nb'], nbUV=opt_net['nbUV'])
    elif which_model == 'ThreeStageCRSR_PseudoYUV420_V1':
        netG = SRResNet_arch.ThreeStageCRSR_PseudoYUV420_V1(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nbY=opt_net['nb'], nbUV=opt_net['nbUV'], sY=opt_net['scaleY'], sUV=opt_net['scaleUV'])
    elif which_model == 'ThreeStageCRSR_PseudoYUV420_V2':
        netG = SRResNet_arch.ThreeStageCRSR_PseudoYUV420_V2(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nbY=opt_net['nb'], nbUV=opt_net['nbUV'], sY=opt_net['scaleY'], sUV=opt_net['scaleUV'])
    elif which_model == 'ThreeStageYUVLiteDWTv2_444':
        netG = SRResNet_arch.ThreeStageYUVLite_DWT_V2_444(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nbY=opt_net['nb'], nbUV=opt_net['nbUV'])
    elif which_model == 'ThreeStageYUVLiteDWTv2_444_UV':
        netG = SRResNet_arch.ThreeStageYUVLite_DWT_V2_444_UV(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nbY=opt_net['nb'], nbUV=opt_net['nbUV'])
    elif which_model == 'ThreeStageYUVLiteDWTv3_444':
        netG = SRResNet_arch.ThreeStageYUVLite_DWT_V3_444(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nbY=opt_net['nb'], nbUV=opt_net['nbUV'])
    elif which_model == 'ThreeStageYUVLiteDWTv3_444_UV':
        netG = SRResNet_arch.ThreeStageYUVLite_DWT_V3_444_UV(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nbY=opt_net['nb'], nbUV=opt_net['nbUV'])
    elif which_model == 'ThreeStageYUVLiteDWTv2_UV':
        netG = SRResNet_arch.ThreeStageYUVLite_DWT_V2_UV(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nbY=opt_net['nb'], nbUV=opt_net['nbUV'])
    elif which_model == 'ThreeStageYUVLiteDWTv3_UV':
        netG = SRResNet_arch.ThreeStageYUVLite_DWT_V3_UV(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nbY=opt_net['nb'], nbUV=opt_net['nbUV'])
    elif which_model == 'ThreeStageYUVLiteDWTv2_new':
        netG = SRResNet_arch.ThreeStageYUVLite_DWT_V2_new(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nbY=opt_net['nb'], nbUV=opt_net['nbUV'])
    elif which_model == 'ThreeStageYUVLite_420to444':
        netG = SRResNet_arch.ThreeStageYUVLite_420to444(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nbY=opt_net['nb'], nbUV=opt_net['nbUV'])
    elif which_model == 'ThreeStageYUVLite_DWT2_444_jointUV' or  which_model == 'eicci_feb':
        netG = SRResNet_arch.ThreeStageYUVLite_DWT2_444_jointUV(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nbY=opt_net['nb'], nbUV=opt_net['nbUV'])        
        
    elif which_model == 'ThreeStageYUVLite_ConvNext_DWT2_444_jointUV':
        netG = SRResNet_arch.ThreeStageYUVLite_ConvNext_DWT2_444_jointUV(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nbY=opt_net['nb'], nbUV=opt_net['nbUV'])        
        
        
        
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG


# Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


# Define network used for perceptual loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF
