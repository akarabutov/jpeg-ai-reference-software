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
from torch import zeros, tensor, int64, mean, zeros_like, cat
from torch.nn import Conv2d, Fold, Unfold
from torch.nn.functional import pad
from torch.nn.parameter import Parameter
from math import log10
from src.codec.common import Image, determinism_on_eval
from src.codec.entropy_coding import ECModule
from copy import deepcopy
from ..base.interface import FilterBase
from .params import EFElinearParams
from typing import List
##
def clamp(num, min_value, max_value):
    return max(min(num, max_value), min_value)

class EFElinear(FilterBase):
    """
        Implements the filter bank interface containing multiple ICCI
        filters.
    """
    conv, convY = [], []

    def __init__(self, **kwargs):
        super(EFElinear, self).__init__(enable_flag_name="EFE_linear_filter_enabled_flag", **kwargs)

        self.conv.append( Conv2d(4,4,(1,1),padding=0,stride = 1, bias=True, groups=4))
        self.conv.append( Conv2d(4,4,(2,2),padding=0,stride = 1, bias=True, groups=4))
        self.conv.append( Conv2d(4,4,(3,3),padding=0,stride = 1, bias=True, groups=4))
        self.conv.append( Conv2d(4,4,(4,4),padding=0,stride = 1, bias=True, groups=4))

        self.convY.append( Conv2d(4,4,(1,1),padding=0,stride = 1, bias=False, groups=4))
        self.convY.append( Conv2d(4,4,(2,2),padding=0,stride = 1, bias=False, groups=4))
        self.convY.append( Conv2d(4,4,(3,3),padding=0,stride = 1, bias=False, groups=4))
        self.convY.append( Conv2d(4,4,(4,4),padding=0,stride = 1, bias=False, groups=4))
        

        self.onetwothree = tensor([0,1,2,3,4,5,6,7])
        self._params_ = EFElinearParams()
        self.cands = [
                [1, 0, [0, 1, 0, 1]],
                [2, 1, [0, 0.5, 0, 1], [0.5, 1, 0, 1]],
                [2, 2, [0, 1, 0, 0.5], [0, 1, 0.5, 1]],
                [3, 3, [0, 1, 0, 0.33], [0, 1, 0.33, 0.66], [0, 1, 0.66, 1]],
                [3, 4, [0, 0.33, 0, 1], [0.33, 0.66, 0, 1], [0.66, 1, 0, 1]],
                [4, 5, [0, 0.5, 0, 0.5], [0, 0.5, 0.5, 1], [0.5, 1, 0, 0.5],[0.5, 1, 0.5, 1]],
                [6, 6, [0, 0.33, 0, 0.5], [0, 0.33, 0.5, 1], [0.33, 0.66, 0, 0.5],[0.33, 0.66, 0.5, 1],[0.66, 1, 0, 0.5],[0.66, 1, 0.5, 1]],
                [6, 7, [0, 0.5, 0, 0.33], [0.5, 1, 0, 0.33], [0, 0.5, 0.33, 0.66],[0.5, 1, 0.33, 0.66],[0, 0.5, 0.66, 1],[0.5, 1, 0.66, 1]],
        ]
        self.wP = 16 #Weight precision
        self.lossModifier = 0.5
        
        self.fL = int(3)
        self.fL_U = int(3)
        self.fL_V = int(3)
        self.filters = {'Y':[],'U':[],'V':[],'U2':[],'V2':[],'candY':-1,'candU':-1,'candV':-1, 'mean1':0,'mean2':0}
        self.bSize = 64
        self.DCT_IF_4TAP = tensor([
        [[[ 0.0,  0,  0, 0],
          [ 0,  0,  0, 0],
          [0,  0,  0, 0],
          [0,  0,  0, 0]]],


        [[[ 0,  0,  0, 0],
          [ -0.0625,  -0.4375,  0.5625, -0.0625],
          [0,  0,  0, 0],
          [0,  0,  0, 0]]],


        [[[ 0,  -0.0625,  0, 0],
          [ 0,  -0.4375,  0, 0],
          [0,   0.5625,  0, 0],
          [0,  -0.0625,  0, 0]]],


        [[[	0.00390625,	-0.03515625,	-0.03515625,	0.00390625	],
        [	-0.03515625,	-0.68359375,	0.31640625,	-0.03515625	],
        [	-0.03515625,	0.31640625,	0.31640625,	-0.03515625	],
        [	0.00390625,	-0.03515625,	-0.03515625,	0.00390625	]]]
        ])

        self.DCT_IF_4TAP_444 = tensor([
        [[[ 0.0,  0,  0, 0],
          [ 0,  0,  0, 0],
          [0,  0,  0, 0],
          [0,  0,  0, 0]]],


        [[[ 0.0,  0,  0, 0],
          [ 0,  0,  0, 0],
          [0,  0,  0, 0],
          [0,  0,  0, 0]]],

        [[[ 0.0,  0,  0, 0],
          [ 0,  0,  0, 0],
          [0,  0,  0, 0],
          [0,  0,  0, 0]]],


        [[[ 0.0,  0,  0, 0],
          [ 0,  0,  0, 0],
          [0,  0,  0, 0],
          [0,  0,  0, 0]]]
        ])
        self.encoder_output = []
    
    def cal_parameter(self, s_ver, s_hor):
        self.o_ver = 2 if s_ver==1 else 1
        self.o_hor = 2 if s_hor==1 else 1
        self.out_format = Image.get_format_from_subsampling(s_ver, s_hor)


    @determinism_on_eval
    def integerize(self,a):
        add = 2**(self.wP-5)
        maxx = 2**(self.wP) - 1
        return(int(clamp(round(a.item()*add) + maxx//2,0,maxx)))

    def pixelShuffleGeneral (self,tensor, LumaH, LumaW, o_ver=2,o_hor=2):
        if o_ver == 1 and o_hor == 1:
            out_tensor = tensor[:,0:1,:,:]
        
        if o_ver == 2 and o_hor == 1:
            chromaH = tensor.shape[2]*2
            chromaW = tensor.shape[3]
            out_tensor = torch.zeros([1,1,chromaH,chromaW], dtype=tensor.dtype, device=tensor.device)
            out_tensor[:,:,0::2,:]=tensor[:,0:1,:,:]
            out_tensor[:,:,1::2,:]=tensor[:,2:3,:,:]
            out_tensor = out_tensor[:,:,:LumaH,:]

        if o_ver == 1 and o_hor == 2:
            chromaH = tensor.shape[2]
            chromaW = tensor.shape[3]*2
            out_tensor = torch.zeros([1,1,chromaH,chromaW], dtype=tensor.dtype, device=tensor.device)
            out_tensor[:,:,:,0::2]=tensor[:,0:1,:,:]
            out_tensor[:,:,:,1::2]=tensor[:,1:2,:,:]
            out_tensor = out_tensor[:,:,:,:LumaW]
        
        if o_ver == 2 and o_hor == 2:
            chromaH = tensor.shape[2]*2
            chromaW = tensor.shape[3]*2
            out_tensor = torch.zeros([1,1,chromaH,chromaW], dtype=tensor.dtype, device = tensor.device)
            out_tensor[:,:,0::2,0::2]=tensor[:,0:1,:,:]
            out_tensor[:,:,0::2,1::2]=tensor[:,1:2,:,:]
            out_tensor[:,:,1::2,0::2]=tensor[:,2:3,:,:]
            out_tensor[:,:,1::2,1::2]=tensor[:,3:4,:,:]
            out_tensor = out_tensor[:,:,:LumaH,:LumaW]
        return out_tensor

    def pixelUnshuffleGeneral (self,tensor, scale_ver=2,scale_hor=2):
        if scale_ver == 1 and scale_hor == 1:
            out_tensor = torch.cat (
                (tensor.clone(), 
                 tensor.clone(),
                 tensor.clone(), 
                 tensor.clone()
                 )
                 ,dim=1)
        
        if scale_ver == 2 and scale_hor == 1:
            tensor_pad = pad(tensor,(0,0,0,1),mode='replicate') if tensor.shape[2]%2 == 1 else tensor #vertical padding
            out_tensor = torch.cat (
                (
                 tensor_pad[:,:,0::2,:],
                 tensor_pad[:,:,0::2,:], 
                 tensor_pad[:,:,1::2,:],
                 tensor_pad[:,:,1::2,:],
                )
                ,dim=1)
            
        if scale_ver == 1 and scale_hor == 2:
            tensor_pad = pad(tensor,(0,1,0,0),mode='replicate') if tensor.shape[3]%2 == 1 else tensor #horizontal padding
            out_tensor = torch.cat (
                (tensor_pad[:,:,:,0::2], 
                 tensor_pad[:,:,:,1::2],
                 tensor_pad[:,:,:,0::2], 
                 tensor_pad[:,:,:,1::2]
                 )
                 ,dim=1)
        
        if scale_ver == 2 and scale_hor == 2:
            tensor_pad = tensor.clone()
            if tensor.shape[3]%2 == 1:
                tensor_pad = pad(tensor_pad,(0,1,0,0),mode='replicate') #horizontal padding
            if tensor.shape[2]%2 == 1:
                tensor_pad = pad(tensor_pad,(0,0,0,1),mode='replicate') #vertical padding
            out_tensor = torch.cat (
                (tensor_pad[:,:,0::2,0::2], 
                 tensor_pad[:,:,0::2,1::2],
                 tensor_pad[:,:,1::2,0::2], 
                 tensor_pad[:,:,1::2,1::2]
                 )
                 ,dim=1)
        return out_tensor
    
    def integerizeTensor(self,tensor):
        dumm = zeros((tensor.shape[0],tensor.shape[1],tensor.shape[2],tensor.shape[3]),dtype=int64)
        for k in range (tensor.shape[0]):
            for l in range (tensor.shape[1]):
                for i in range (tensor.shape[2]):
                    for j in range (tensor.shape[3]):
                        dumm[k,l,i,j] = self.integerize(tensor[k,l,i,j])
        return dumm


    def deinteger(self,a):
        add = 2**(self.wP-5)
        maxx = 2**(self.wP) - 1
        if isinstance(a, int):
            return float(a - maxx//2)/add
        elif isinstance(a, list):
            a = tensor(a)
            return (a - maxx//2).float()/add
        else:
            return (a - maxx//2).float()/add
    
    def compress(self, imgs: List[Image], org_img_i: Image, *args, **kwargs) -> List[Image]:
        rec = imgs[0]
        self.s_ver = self.get_owner_param('s_ver')
        self.s_hor = self.get_owner_param('s_hor')
        self.c_ver = self.get_owner_param('c_ver')
        self.c_hor = self.get_owner_param('c_hor')
        self.scale_ver = 3 - self.c_ver/self.s_ver
        self.scale_hor = 3 - self.c_hor/self.s_hor
        self.cal_parameter(self.s_ver, self.s_hor)
        torch.cuda.empty_cache()
        hh,ww = org_img_i.get_component('a').shape[-2:]
        model_id = self.get_base_model_id()
        img = org_img_i.clone()
        img.to_YUV_()
        img.convert_range_(rec.data_range)
        self.c = cat((img.get_component('b').clone(), img.get_component('c').clone()), dim=1)
        self.luma = img.get_component('a').clone()

        h, w = self.luma.shape[-2:]
        rec_upsample = None
        self.filtersBest2 = deepcopy(self.filters)
        if w*h<4000*4000 and (not self.DCTIF_only) and self.owner.EFEnonlinear.enabled:
            rec_upsample, self.filtersBest2 = self.SplitDecide(rec,[1], self.cands[0:1])
        filtL, cands = [1,2,3,4], self.cands

        if w*h<=1e6:
            cands = self.cands[0:3] if model_id < 2 else self.cands[4:]
            filtL = [1, 2] if model_id == 0 else [3, 4]
        elif w*h<=4e6:
            cands = self.cands if model_id < 2 else self.cands[4:]
            if model_id == 0:
                filtL = [1, 2, 3] 
            elif model_id == 1:
                filtL = [2, 3, 4]
            else:
                filtL = [3, 4]
        elif w*h<=9e6:
            cands = self.cands[3:] if model_id < 2 else self.cands[6:]
            if model_id == 0:
                filtL = [1, 2, 3] 
            elif model_id == 1:
                filtL = [3, 4]
            else:
                filtL = [4]
        elif w*h<16e6:
            cands = self.cands[5:] if model_id < 2 else self.cands[6:]
            filtL = [1, 2, 3] if model_id == 0 else [4]
        else:
            cands = self.cands[6:]
            filtL = [3, 4] if model_id == 0 else [4]

        if self.DCTIF_only:
            filtL, cands = [1], self.cands[0:1]
        rec, self.filtersBest= self.SplitDecide(rec,filtL, cands)
        if self.DCTIF_only:
            self.filtersBest2 = {'Y':[],'U':[],'V':[],'U2':[],'V2':[],'candY':-1,'candU':-1,'candV':-1, 'mean1':self.filtersBest2['mean1'],'mean2':self.filtersBest2['mean2']}
            self.filtersBest = {'Y':[],'U':[],'V':[],'U2':[],'V2':[],'candY':-1,'candU':-1,'candV':-1, 'mean1':self.filtersBest2['mean1'],'mean2':self.filtersBest2['mean2']}
        torch.cuda.empty_cache()
        ans = [rec, rec_upsample]
        if len(imgs) > 2:
            ans += imgs[2:]        
        return ans

    @determinism_on_eval
    def decompress(self, imgs: List[Image], *args, **kwargs) -> Image:
        img = imgs[0]
        self.doPixelUnShuffle = True
        ans = self.SplitApply(img,self.cands,self.filtersBest)
        ans_upsample = None
        if (self.filtersBest2['candU'] > -1 or self.filtersBest2['candV'] > -1) and self.owner.EFEnonlinear.enabled:
            if self.owner.EFEnonlinear.mask1.shape[2] > 0 or self.owner.EFEnonlinear.mask1.shape[3] > 0: 
                self.doPixelUnShuffle = False
                ans_upsample = self.SplitApply(img, self.cands, self.filtersBest2, None, None)

        ans = [ans, ans_upsample] 
        if len(imgs) > 2:
            ans += imgs[2:]
        return ans
    
    
    def SplitDecide(self ,img: Image, filterLengths, candidatesList):
        #input is yuv420
        rec_Y = img.get_component('a').clone()
        rec_U = img.get_component('b').clone()
        rec_V = img.get_component('c').clone()
        mean1 = round(mean(rec_U).item()*100)/100
        mean2 = round(mean(rec_V).item()*100)/100 
        h, w = rec_Y.shape[-2:]
        U_org_pad = self.pixelUnshuffleGeneral(self.c[:,0:1],3-self.s_ver,3-self.s_hor)
        V_org_pad = self.pixelUnshuffleGeneral(self.c[:,1:2],3-self.s_ver,3-self.s_hor)
        rec_U_pad = self.pixelUnshuffleGeneral(rec_U,3-self.s_ver,3-self.s_hor)
        rec_U_pad = pad(rec_U_pad,(2,2,2,2),mode='replicate')
        rec_V_pad = self.pixelUnshuffleGeneral(rec_V,3-self.s_ver,3-self.s_hor)
        rec_V_pad = pad(rec_V_pad,(2,2,2,2),mode='replicate')
        rec_Y_pad = self.pixelUnshuffleGeneral(rec_Y,2,2)

        rec_Y_pad = pad(rec_Y_pad,(2,2,2,2),mode='replicate')

        rec_U_new = zeros_like(U_org_pad)
        rec_V_new = zeros_like(V_org_pad)
        s = rec_U_new.shape[-2:]

        betalist = [0.002, 0.012, 0.075, 0.5]
        beta = betalist[self.get_base_model_id()] 

        conv_UV_fix = Conv2d(4,4,(4,4),padding=0,stride = 1, groups=4, bias=False,device=rec_Y.device)
        if self.scale_hor == 2 and self.scale_ver == 2: 
            conv_UV_fix.weight = Parameter(self.DCT_IF_4TAP_444.to(rec_Y.device))
        else:
            conv_UV_fix.weight = Parameter(self.DCT_IF_4TAP.to(rec_Y.device))
        upsampled_U = conv_UV_fix(rec_U_pad[:,:,1:,1:] - mean1)
        upsampled_V = conv_UV_fix(rec_V_pad[:,:,1:,1:] - mean2)
        
        
        def searchCore(fl, cand):
            numSplit = cand[0]
            filtersY_U, filtersY_V, filtersV, filtersU = [],[],[],[]

            for num in range(numSplit):
                split = cand[num+2]
                
                padd_down = fl//2
                padd_up = padd_down if fl%2 == 1 else padd_down-1
                sli_l_h = slice(2 - padd_up+ round(split[0]*(s[0]+31)/32)*32, 2 + padd_down + min(round(split[1]*(s[0]+31)/32)*32, s[0]))
                sli_l_v = slice(2 - padd_up+ round(split[2]*(s[1]+31)/32)*32, 2 + padd_down + min(round(split[3]*(s[1]+31)/32)*32, s[1]))

                sli_l_h_org = slice(round(split[0]*(s[0]+31)/32)*32,min(round(split[1]*(s[0]+31)/32)*32, s[0]))
                sli_l_v_org = slice(round(split[2]*(s[1]+31)/32)*32,min(round(split[3]*(s[1]+31)/32)*32, s[1]))

                weightsY, weightsV = self.LumaAidedUpsampler_encoder(rec_Y_pad[:,:,sli_l_h,sli_l_v].clone(), rec_U_pad[:,:,sli_l_h,sli_l_v].clone(), rec_V_pad[:,:,sli_l_h,sli_l_v].clone(), U_org_pad[:,:,sli_l_h_org,sli_l_v_org], V_org_pad[:,:,sli_l_h_org,sli_l_v_org], fl, mean1, mean2, upsampled_U[:,:,sli_l_h_org,sli_l_v_org], upsampled_V[:,:,sli_l_h_org,sli_l_v_org])
                filtersY_U.append(self.integerizeTensor(weightsY[0]))
                filtersY_V.append(self.integerizeTensor(weightsY[1]))
                filtersU.append(self.integerizeTensor(weightsV[0]))
                filtersV.append(self.integerizeTensor(weightsV[1]))
                
                self.weightsUV = self.deinteger(filtersU[num]).to(rec_Y_pad.device)
                self.weightsY_UV = self.deinteger(filtersY_U[num]).to(rec_Y_pad.device)
                start = 1
                end = 2
                if self.scale_hor == 2 and self.scale_ver == 2:
                    fL = 0 if self.weightsUV is None else self.weightsUV.shape[3]
                    if fL == 3: 
                        start = 1
                        end = 1
                    if fL == 2: 
                        start = 0
                        end = 1
                    if fL == 1: 
                        start = 0
                        end = 0
                sli_l_h = slice(2 - start+ round(split[0]*(s[0]+31)/32)*32, 2 + end + min(round(split[1]*(s[0]+31)/32)*32, s[0]))
                sli_l_v = slice(2 - start+ round(split[2]*(s[1]+31)/32)*32, 2 + end + min(round(split[3]*(s[1]+31)/32)*32, s[1]))

                self.meanUV = mean1
                adu = self.LumaAidedUpsampler_apply(rec_Y_pad[:,:,sli_l_h,sli_l_v].clone(), rec_U_pad[:,:,sli_l_h,sli_l_v].clone())

                self.weightsUV = self.deinteger(filtersV[num]).to(rec_Y_pad.device)
                self.weightsY_UV = self.deinteger(filtersY_V[num]).to(rec_Y_pad.device)
                self.meanUV = mean2
                adv = self.LumaAidedUpsampler_apply(rec_Y_pad[:,:,sli_l_h,sli_l_v].clone(), rec_V_pad[:,:,sli_l_h,sli_l_v].clone())

                rec_V_new[:,:,sli_l_h_org,sli_l_v_org] = adv
                rec_U_new[:,:,sli_l_h_org,sli_l_v_org] = adu
            return rec_U_new, rec_V_new, filtersY_U, filtersY_V, filtersV, filtersU

        rec_U, rec_V, filtersY_U, filtersY_V, filtersV, filtersU = searchCore(1, self.cands[0])
        rec_U_best, rec_V_best, filtersU2_best, filtersV2_best, filtersV_best, filtersU_best = rec_U.clone(),rec_V.clone(), deepcopy(filtersY_U), deepcopy(filtersY_V), deepcopy(filtersV), deepcopy(filtersU)
        best_cand_u_idx = 0
        best_cand_v_idx = 0

        before_u = 10*log10(255**2/mean((U_org_pad - rec_U_best)**2))
        before_v = 10*log10(255**2/mean((V_org_pad - rec_V_best)**2))
        numFilters = 2 if (self.scale_ver == 2 and self.scale_hor==2) else 5
        loss_best_u = beta * (before_u) - (numFilters*(1**2)*1*self.lossModifier)/(h*w)
        loss_best_v = beta * (before_v) - (numFilters*(1**2)*1*self.lossModifier)/(h*w)
        breakiteration = False
        for fl in filterLengths:
            if breakiteration:
                break
            for cand in candidatesList:
                if fl == 1 and cand[0] == 1:
                    continue
                rec_U, rec_V, filtersY_U, filtersY_V, filtersV, filtersU = searchCore(fl, cand)
                
                after_u = 10*log10(255**2/mean((U_org_pad - rec_U)**2))
                after_v = 10*log10(255**2/mean((V_org_pad - rec_V)**2))
                loss_u = beta * (after_u) - (5*(fl**2)*self.wP*cand[0]*self.lossModifier)/(h*w)
                loss_v = beta * (after_v) - (5*(fl**2)*self.wP*cand[0]*self.lossModifier)/(h*w)

                if loss_best_u < loss_u:
                   
                    rec_U_best = rec_U.clone()
                    loss_best_u = loss_u
                    best_cand_u_idx = cand[1]
                    filtersU_best = deepcopy(filtersU)
                    filtersU2_best = deepcopy(filtersY_U)
                if loss_best_v < loss_v:

                    rec_V_best = rec_V.clone()
                    loss_best_v = loss_v
                    best_cand_v_idx = cand[1]
                    filtersV_best = deepcopy(filtersV)
                    filtersV2_best = deepcopy(filtersY_V)
                
                
                if fl == 2 and filtersU_best[0].shape[3] == 1 and filtersV_best[0].shape[3] == 1 and best_cand_v_idx == 0 and best_cand_u_idx == 0:
                    breakiteration = True
                    break
                if fl == 3 and filtersU_best[0].shape[3] < 2 and filtersV_best[0].shape[3] < 2:
                    breakiteration = True
                    break
                if fl == 4 and filtersU_best[0].shape[3] < 3 and filtersV_best[0].shape[3] < 3:
                    breakiteration = True
                    break
        rec_U_best = self.pixelShuffleGeneral(rec_U_best,o_ver=self.o_ver,o_hor=self.o_hor, LumaH=rec_Y.shape[2], LumaW=rec_Y.shape[3])
        rec_V_best = self.pixelShuffleGeneral(rec_V_best,o_ver=self.o_ver,o_hor=self.o_hor, LumaH=rec_Y.shape[2], LumaW=rec_Y.shape[3])
        ans = Image.create_from_tensors(rec_Y,
                                rec_U_best,
                                rec_V_best,
                                img.data_range,
                                format=self.out_format,
                                bit_depth=img.bit_depth,
                                color_space='yuv')
        filters = {'mean1':mean1, 'mean2':mean2, 'Y':[],'U':filtersU_best,'V':filtersV_best,'candY':-1,'candU':best_cand_u_idx,'candV':best_cand_v_idx, 'U2':filtersU2_best, 'V2':filtersV2_best}
        return ans , filters

    def SplitApply(self ,img: Image, candidatesList, filtersList, m1 = None, m2 = None):
        #input is yuv420, 444 or 422
        if self.doPixelUnShuffle:
            self.rec_Y = img.get_component('a').clone()
            self.rec_U = img.get_component('b').clone()
            self.rec_V = img.get_component('c').clone()

            self.rec_Y_pad = self.pixelUnshuffleGeneral(self.rec_Y,2,2)
            self.rec_U_pad = self.pixelUnshuffleGeneral(self.rec_U,3-self.s_ver,3-self.s_hor)
            self.rec_V_pad = self.pixelUnshuffleGeneral(self.rec_V,3-self.s_ver,3-self.s_hor)
            self.rec_U_pad = pad(self.rec_U_pad,(2,2,2,2),mode='replicate')
            self.rec_V_pad = pad(self.rec_V_pad,(2,2,2,2),mode='replicate')

            self.rec_U_new = zeros_like(self.rec_Y_pad)
            self.rec_V_new = zeros_like(self.rec_Y_pad)
            self.rec_Y_pad = pad(self.rec_Y_pad,(2,2,2,2),mode='replicate')
        s = self.rec_U_new.shape[-2:]

        for j in range (2):
            if j == 0:
                UV, UV2, mean, candID = "U", "U2", "mean1", "candU"
                rec, rec_new = self.rec_U_pad, self.rec_U_new.clone()
                if (not m1 == None) and m1.shape[2] == 0:
                    rec_U_new = self.rec_U
                    continue
            else:
                UV, UV2, mean, candID = "V", "V2", "mean2", "candV"
                rec, rec_new = self.rec_V_pad, self.rec_V_new.clone()
                if (not m2 == None) and m2.shape[2] == 0:
                    rec_V_new = self.rec_V
                    continue

            cand = candidatesList[filtersList[candID]]
            numSplit = cand[0]
            for num in range(numSplit):
                split = cand[num+2]
                start = 1
                end = 2
                self.weightsUV,self.weightsY_UV = None, None
                if len(filtersList[UV]) > 0:
                    self.weightsUV = self.deinteger(filtersList[UV][num]).to(self.rec_Y_pad.device)
                if len(filtersList[UV2]) > 0:
                    self.weightsY_UV = self.deinteger(filtersList[UV2][num]).to(self.rec_Y_pad.device)
                if self.scale_hor == 2 and self.scale_ver == 2:
                    fL = 0 if self.weightsUV is None else self.weightsUV.shape[3]
                    if fL == 3: 
                        start = 1
                        end = 1
                    if fL == 2: 
                        start = 0
                        end = 1
                    if fL == 1: 
                        start = 0
                        end = 0
                sli_l_h = slice(2 - start+ round(split[0]*(s[0]+31)/32)*32, 2 + end + min(round(split[1]*(s[0]+31)/32)*32, s[0]))
                sli_l_v = slice(2 - start+ round(split[2]*(s[1]+31)/32)*32, 2 + end + min(round(split[3]*(s[1]+31)/32)*32, s[1]))
                sli_l_h_org = slice(round(split[0]*(s[0]+31)/32)*32, min(round(split[1]*(s[0]+31)/32)*32, s[0]))
                sli_l_v_org = slice(round(split[2]*(s[1]+31)/32)*32, min(round(split[3]*(s[1]+31)/32)*32, s[1]))

                self.meanUV = filtersList[mean]
                adu = self.LumaAidedUpsampler_apply(self.rec_Y_pad[:,:,sli_l_h,sli_l_v], rec[:,:,sli_l_h,sli_l_v])
                rec_new[:,:,sli_l_h_org,sli_l_v_org] = adu
            if j == 0:
                rec_U_new = self.pixelShuffleGeneral(rec_new,o_ver=self.o_ver,o_hor=self.o_hor, LumaH=self.rec_Y.shape[2]>>(self.s_ver-1), LumaW=self.rec_Y.shape[3]>>(self.s_hor-1))
            else:
                rec_V_new = self.pixelShuffleGeneral(rec_new,o_ver=self.o_ver,o_hor=self.o_hor, LumaH=self.rec_Y.shape[2]>>(self.s_ver-1), LumaW=self.rec_Y.shape[3]>>(self.s_hor-1))
        ans = Image.create_from_tensors(self.rec_Y,
                                    rec_U_new,
                                    rec_V_new,
                                    img.data_range,
                                    bit_depth=img.bit_depth,
                                    format=self.out_format,
                                    color_space='yuv')
        return ans


    def LumaAidedUpsampler_encoder(self,rec_Y,rec_U,rec_V, U_org, V_org, fL_V, mean1, mean2, upsampled_U, upsampled_V):
        def linearSolve(A,B):
            max_samples = 2000000
            subsample = max(round(A.shape[1]/max_samples),1)
            X = torch.zeros((1,A.shape[2],1),device=A.device)
            for i in range(0,subsample*4,4):
                X += torch.linalg.lstsq(A[:,i::subsample*4,:], B[:,i::subsample*4,:]).solution
            return X/subsample
        unfold_UV = Unfold((fL_V,fL_V),padding=0)
        unfold_UV2 = Unfold((1,1),padding=0)
        fold_UV = Fold(output_size=(fL_V, fL_V), kernel_size=(1, 1))
        padd_down = fL_V//2
        padd_up = padd_down if fL_V%2 == 1 else padd_down-1
        
        weightsY_UV = []
        weightsUV = []
        for j in range (2):
            if j ==0:
                rec_UV, meanUV, org_UV, upsampled_UV = rec_U, mean1, U_org, upsampled_U
            else:
                rec_UV, meanUV, org_UV, upsampled_UV = rec_V, mean2, V_org, upsampled_V
            

            toCollect, toCollect2 = [], []
            for i in range (4):
                if i == 0 or not (self.scale_hor == 2 and self.scale_ver==2): 
                    A1 = unfold_UV(rec_UV[:,i:i+1] - meanUV).permute(0,2,1)
                    A2 = unfold_UV(rec_Y[:,i:i+1]).permute(0,2,1)
                    A = cat((A1,A2),dim=2)
                    B = unfold_UV2(org_UV[:,i:i+1] - pad(rec_UV[:,i:i+1], (-padd_up, -padd_down, -padd_up, -padd_down,)) - upsampled_UV[:,i:i+1]).permute(0,2,1)
                    X = linearSolve(A, B)
                toCollect.append(fold_UV(X[:,:X.shape[1]//2,:].permute(0,2,1)))
            weightsV = cat(toCollect, dim=0)
            weightsY = fold_UV(X[:,X.shape[1]//2:,:].permute(0,2,1))
            #refine the luma filter.
            if not (self.scale_hor == 2 and self.scale_ver==2): 
                conv_UV = Conv2d(4,4,(fL_V,fL_V),padding=0,stride = 1, groups = 4, bias=False,device=rec_Y.device)
                conv_UV.weight = Parameter(weightsV)
                rec_add =  pad(rec_UV, (-padd_up, -padd_down, -padd_up, -padd_down)) + conv_UV(rec_UV - meanUV) + upsampled_UV
                
                A, B = [],[]
                for i in range (4):
                    A.append(unfold_UV(rec_Y[:,i:i+1]).permute(0,2,1))
                    B.append(unfold_UV2(org_UV[:,i:i+1] - rec_add[:,i:i+1]).permute(0,2,1))
                X = linearSolve(cat(A,dim=1), cat(B,dim=1))
                weightsY = fold_UV(X.permute(0,2,1))
            weightsY_UV.append(weightsY)
            
            weightsUV.append(weightsV)

        return weightsY_UV, weightsUV

    def LumaAidedUpsampler_apply(self,rec_Y,rec_UV):
        fL = 0 if self.weightsUV is None else self.weightsUV.shape[3]
        if self.scale_hor == 2 and self.scale_ver == 2:
            self.conv[fL-1].to(self.weightsY_UV.device)
            self.convY[fL-1].to(self.weightsY_UV.device)
            conv = self.conv[fL-1]
            convY = self.convY[fL-1]
            conv.bias[:] = self.meanUV
            if self.weightsY_UV is not None:
                for i in range(4):
                    convY.weight[i:i+1,0:1,:,:]  = Parameter(self.weightsY_UV[:,:,:,:])
                    conv.weight[i:i+1,0:1,:,:]  = Parameter(self.weightsUV[i:i+1,:,:,:])
            conv.weight[:,:,(fL-1)//2:(fL-1)//2 + 1,(fL-1)//2:(fL-1)//2 + 1] += 1
        else:
            conv = Conv2d(4,4,(4,4),padding=0,stride = 1, bias=True,device=rec_Y.device, groups=4)
            convY = Conv2d(4,4,(4,4),padding=0,stride = 1, bias=False,device=rec_Y.device, groups=4)
            conv.weight.data.zero_()
            convY.weight.data.zero_()
            conv.bias[:] = self.meanUV
            if self.weightsY_UV is not None:
                for i in range(4):
                    convY.weight[i:i+1,0:1,1 - (fL-1)//2 : 2+fL//2,1 - (fL-1)//2 : 2+fL//2]  = Parameter(self.weightsY_UV[:,:,:,:])
                    conv.weight[i:i+1,0:1,1 - (fL-1)//2 : 2+fL//2,1 - (fL-1)//2 : 2+fL//2]  = Parameter(self.weightsUV[i:i+1,:,:,:])
            if self.scale_ver == 2 and self.scale_hor == 2:
                conv.weight += Parameter(self.DCT_IF_4TAP_444.to(rec_UV.device))
            else:
                conv.weight += Parameter(self.DCT_IF_4TAP.to(rec_UV.device))
            conv.weight[:,:,1:2,1:2] += 1
        return  conv(rec_UV - self.meanUV) + convY(rec_Y)


    def decode_header(self, ec: ECModule):
        self.s_ver = self.get_owner_param('s_ver')
        self.s_hor = self.get_owner_param('s_hor')
        self.c_ver = self.get_owner_param('c_ver')
        self.c_hor = self.get_owner_param('c_hor')
        self.scale_ver = 3 - self.c_ver/self.s_ver
        self.scale_hor = 3 - self.c_hor/self.s_hor
        self.cal_parameter(self.s_ver, self.s_hor)
        mean1 = float(int(ec.decode([1], max_symbol_value=2**15-1, name='B1[0]')))/100
        mean2 = float(int(ec.decode([1], max_symbol_value=2**15-1, name='B1[1]')))/100
        self.filtersBest2 = self.decode_filters(ec)
        self.filtersBest = self.decode_filters(ec)
        self.filtersBest['mean1'] = self.filtersBest2['mean1'] = mean1
        self.filtersBest['mean2'] = self.filtersBest2['mean2'] = mean2



    def encode_header(self, ec: ECModule):
        self.maxxx = int(2**self.wP - 1)
        self.minnn = int(0)
        ec.encode((int(round(self.filtersBest['mean1']*100))), max_symbol_value=2**15-1, name='B1[0]')
        ec.encode((int(round(self.filtersBest['mean2']*100))), max_symbol_value=2**15-1, name='B1[1]')
        self.logger.debug(f'first filter set')
        self.encode_filters(ec, self.filtersBest2)

        self.minnn = int(2**self.wP - 1)
        self.maxxx = int(0)
        if self.filtersBest['candU'] > -1:
            for k in range(self.cands[self.filtersBest['candU']][0]):
                self.minnn = min(torch.min(self.filtersBest['U'][k]).item(),self.minnn)
                self.maxxx = max(torch.max(self.filtersBest['U'][k]).item(),self.maxxx)
                self.minnn = min(torch.min(self.filtersBest['U2'][k]).item(),self.minnn)
                self.maxxx = max(torch.max(self.filtersBest['U2'][k]).item(),self.maxxx)
                
        if self.filtersBest['candV'] > -1:
            for k in range(self.cands[self.filtersBest['candV']][0]):
                self.minnn = min(torch.min(self.filtersBest['V'][k]).item(),self.minnn)
                self.maxxx = max(torch.max(self.filtersBest['V'][k]).item(),self.maxxx)
                self.minnn = min(torch.min(self.filtersBest['V2'][k]).item(),self.minnn)
                self.maxxx = max(torch.max(self.filtersBest['V2'][k]).item(),self.maxxx)

        self.logger.debug(f'second filter set')
        self.encode_filters(ec, self.filtersBest)

    def encode_filters(self, ec: ECModule, filters):
        ec.encode(int(filters['candU']+1), max_symbol_value=15, name='best_cand_idx[0]')
        ec.encode(int(filters['candV']+1), max_symbol_value=15, name='best_cand_idx[1]')
        
        if filters['candU']>-1:
            ec.encode(int(filters['U'][0].shape[2]), max_symbol_value=9, name='fL[0]')
            
        if filters['candV']>-1:
            ec.encode(int(filters['V'][0].shape[2]), max_symbol_value=9, name='fL[1]')

        self.logger.debug(f'best_cand_u_idx {filters["candU"]}')
        self.logger.debug(f'fL_U {filters["U"][0].shape[2]}') if filters['candU']>-1 else self.logger.debug(f'fL_U 0')
        self.logger.debug(f'best_cand_v_idx {filters["candV"]}')
        self.logger.debug(f'fL_V {filters["V"][0].shape[2]}') if filters['candV']>-1 else self.logger.debug(f'fL_V 0')
        self.maxSymbol = abs(int(self.maxxx) - int(self.minnn))
        ec.encode(self.minnn, max_symbol_value=2**self.wP-1, name='minSymbol')
        ec.encode(self.maxSymbol, max_symbol_value=2**self.wP-1, name='maxSymbol')

        def encode_one_filter(filters, name):
            for filt in filters:
                if self.scale_hor == 2 and self.scale_ver == 2:
                    ec.encode(filt[0:1,:,:,:] - self.minnn, max_symbol_value=self.maxSymbol, name=name)
                else:
                    ec.encode(filt - self.minnn, max_symbol_value=self.maxSymbol, name=name)
                
        if filters['candU'] > -1:
            encode_one_filter(filters['U'], 'weightsU')
            encode_one_filter(filters['U2'], 'weightsU2')

        if filters['candV'] > -1:
            encode_one_filter(filters['V'], 'weightsV')
            encode_one_filter(filters['V2'], 'weightsV2')



    def decode_filters(self,ec: ECModule):

        def decode_one_filter(numFilter, shape, name):
            
            shapeSingle = (1,shape[1],shape[2],shape[3])
            if self.scale_hor == 2 and self.scale_ver == 2: #TODO: this modification needs to be updated in the spec. if coding mode is 4:4:4, only 1 chrome filter is parsed, and it is used in all 4 planes.
                wU = zeros(shapeSingle,dtype=int64)
            else:
                wU = zeros(shape,dtype=int64)
            filters = []
            for _ in range(numFilter):
                if self.scale_hor == 2 and self.scale_ver == 2 and shape[0]== 4:
                    wU = ec.decode(shapeSingle, max_symbol_value=self.maxSymbol, name=name) + self.minnn
                    wU = torch.cat((wU.clone(),wU.clone(),wU.clone(),wU.clone()),dim=0)
                else:
                    wU = ec.decode(shape, max_symbol_value=self.maxSymbol, name=name) + self.minnn
                filters.append(wU.clone())
            return filters


        filters = deepcopy(self.filters)
        logger = self.logger
        filters['candU'] = int(ec.decode([1], max_symbol_value=15, name='best_cand_idx[0]'))-1
        filters['candV'] = int(ec.decode([1], max_symbol_value=15, name='best_cand_idx[1]'))-1
        logger.debug(f'best_cand_u_idx: {filters["candU"]}')
        logger.debug(f'best_cand_v_idx: {filters["candV"]}')

        if filters['candU']>-1:
            fL_U = int(ec.decode([1], max_symbol_value=9, name='fL[0]'))
        if filters['candV']>-1:
            fL_V = int(ec.decode([1], max_symbol_value=9, name='fL[1]'))
        
        self.NonlinearFilterWeights_U = []
        self.NonlinearFilterWeights_V = []

        self.minnn = int(ec.decode([1], max_symbol_value=2**self.wP-1, name='minSymbol'))
        self.maxSymbol = int(ec.decode([1], max_symbol_value=2**self.wP-1, name='maxSymbol'))
        


        if filters["candU"]>-1:
            filters['U'] = decode_one_filter(self.cands[filters["candU"]][0],(4,1,fL_U,fL_U), name='weightsU' )
            filters['U2'] = decode_one_filter(self.cands[filters["candU"]][0],(1,1,fL_U,fL_U), name='weightsU2' )

        if filters["candV"]>-1:
            filters['V'] = decode_one_filter(self.cands[filters["candV"]][0],(4,1,fL_V,fL_V),name='weightsV' )
            filters['V2'] = decode_one_filter(self.cands[filters["candV"]][0],(1,1,fL_V,fL_V),name='weightsV2' )

        filters['Y'] = []
        return filters
    
    