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
from einops import repeat
import torch
from typing import List
from torch import zeros, tensor, int64, mean, cat
from torch import float as torch_float
from torch.nn import AvgPool2d, Conv2d, ReLU, Fold, Unfold
from torch.nn.functional import pad
from math import log10
from src.codec.common import Image, determinism_on_eval
from src.codec.entropy_coding import ECModule
from ..base import FilterBase
##
def clamp(num, min_value, max_value):
    return max(min(num, max_value), min_value)


class EFEnonlinear(FilterBase):
    """
        Implements the filter bank interface containing multiple ICCI
        filters.
    """
    def __init__(self, **kwargs):
        super(EFEnonlinear, self).__init__(enable_flag_name="EFE_nonlinear_filter_enabled_flag", **kwargs)
        self.onetwothree = tensor([0,1,2,3,4,5,6,7])
        self.wP = 16 #Weight precision
        self.lossModifier = 1.5
        self.NonlinearFilter_tile_width_base = 1200
        self.NonlinearFilter_tile_height_base = 1200
        
        self.bSize = 64
        self.d_ver = 1
        self.d_hor = 1
        self.hist_split_num = 8

    @determinism_on_eval
    def integerize(self,a):
        add = 2**(self.wP-5)
        maxx = 2**(self.wP) - 1
        return(int(clamp(round(a.item()*add) + (maxx//2),0,maxx)))

    def cal_parameter(self, d_ver, d_hor):
        self.out_format = '444'
        if d_ver == 2 or d_hor == 2:
            self.out_format = '422'
        if d_ver == 2 and d_hor == 2:
            self.out_format = '420'

    def downsample(self, tensor):
        d_ver = self.d_ver
        d_hor = self.d_hor
        tensor_pad = tensor
        if d_ver == 1 and d_hor == 1:
            return tensor_pad
        if d_ver == 2 and d_hor == 2:
            if tensor.shape[2]%2 == 1:
                tensor_pad = pad(tensor,(0,0,0,1),mode='replicate') #vertical padding
            if tensor.shape[3]%2 == 1:
                tensor_pad = pad(tensor,(0,1,0,0),mode='replicate') #horizontal padding
            return tensor_pad[:,:,::2,::2]
        if d_ver == 2 and d_hor == 1:
            if tensor.shape[2]%2 == 1:
                tensor_pad = pad(tensor,(0,0,0,1),mode='replicate') #vertical padding
            return tensor_pad[:,:,::2,:]
        if d_ver == 1 and d_hor == 2:
            if tensor.shape[3]%2 == 1:
                tensor_pad = pad(tensor,(0,1,0,0),mode='replicate') #horizontal padding
            return tensor_pad[:,:,:,::2]
        

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
    
    def calculateOnOff(self, rec: Image, filt:Image, org_img_i:Image):
        org_img_i.to_YUV_()
        if self.d_hor!=2 and self.d_ver!=2:
            org_img_i.to_444_()
        org_img_i.convert_range_(filt.data_range)
        rec.to_YUV_()
        if self.d_hor!=2 and self.d_ver!=2:
            rec.to_444_()
        rec.convert_range_(filt.data_range)


        h, w = rec.get_component('b').shape[-2:]
        h_diff = ((h + self.bSize - 1)//self.bSize)*self.bSize - h
        w_diff = ((w + self.bSize - 1)//self.bSize)*self.bSize - w

        org_img_i.pad_(w_diff,h_diff)
        rec.pad_(w_diff,h_diff)
        filt.pad_(w_diff,h_diff)
        
        recUave = (rec.get_component('b') + filt.get_component('b'))/2
        recVave = (rec.get_component('c') + filt.get_component('c'))/2
        
        pool = AvgPool2d((self.bSize,self.bSize),(self.bSize,self.bSize))
        self.mask1 = pool((org_img_i.get_component('b') - filt.get_component('b'))**2) > pool((org_img_i.get_component('b') - rec.get_component('b'))**2)
        self.mask2 = pool((org_img_i.get_component('c') - filt.get_component('c'))**2) > pool((org_img_i.get_component('c') - rec.get_component('c'))**2)
        self.mask1 = self.mask1*2
        self.mask2 = self.mask2*2

        mask = repeat(self.mask1,
                'a b c d -> a b (c repeat1) (d repeat2)',
                repeat1=self.bSize,
                repeat2=self.bSize).to(rec.device).to(dtype=torch_float)

        
        maskedU = (filt.get_component('b')*(2-mask) + (mask)*rec.get_component('b'))/2
        mask = repeat(self.mask2,
                'a b c d -> a b (c repeat1) (d repeat2)',
                repeat1=self.bSize,
                repeat2=self.bSize).to(rec.device).to(dtype=torch_float)
        maskedV = (filt.get_component('c')*(2-mask) + (mask)*rec.get_component('c'))/2
        mask3 = pool((org_img_i.get_component('b') - maskedU)**2) > pool((org_img_i.get_component('b') - recUave)**2)
        mask4 = pool((org_img_i.get_component('c') - maskedV)**2) > pool((org_img_i.get_component('c') - recVave)**2)
        self.mask1[mask3 ==1] = 1
        self.mask2[mask4 ==1] = 1

    def apply_OnoffSwitch(self, rec: Image, filt:Image):
        if self.d_hor!=2 and self.d_ver!=2:
            rec.to_444_()
        rec.convert_range_(filt.data_range)
        
        h, w = rec.get_component('b').shape[-2:]
        h_diff = ((h + self.bSize - 1)//self.bSize)*self.bSize - h
        w_diff = ((w + self.bSize - 1)//self.bSize)*self.bSize - w
        rec.pad_(w_diff,h_diff)
        filt.pad_(w_diff,h_diff)
        
        filt_U = filt.get_component('b').clone()
        if self.mask1.shape[2] > 0:
            mask = repeat(self.mask1.to(rec.device).to(dtype=torch_float),
                            'a b c d -> a b (c repeat1) (d repeat2)',
                            repeat1=self.bSize,
                            repeat2=self.bSize)
            filt_U = (rec.get_component('b')*(mask) + (2-mask)*filt_U)/2

        
        filt_V = filt.get_component('c').clone()
        if self.mask2.shape[2] > 0:
            mask = repeat(self.mask2.to(rec.device).to(dtype=torch_float),
                            'a b c d -> a b (c repeat1) (d repeat2)',
                            repeat1=self.bSize,
                            repeat2=self.bSize)
            filt_V = (rec.get_component('c')*(mask) + (2-mask)*filt_V)/2
        
        filt_Y = filt.get_component('a')
        if h_diff > 0:
            filt_V = filt_V[:,:,:-h_diff,:]
            filt_U = filt_U[:,:,:-h_diff,:]
            filt_Y = filt_Y[:,:,:-h_diff,:]
        if w_diff > 0:
            filt_V = filt_V[:,:,:,:-w_diff]
            filt_U = filt_U[:,:,:,:-w_diff]
            filt_Y = filt_Y[:,:,:,:-w_diff]
        ans = Image.create_from_tensors(filt_Y,
                                        filt_U,
                                        filt_V,
                                        filt.data_range,
                                        bit_depth=rec.bit_depth,
                                        format=self.out_format,
                                        color_space='yuv')
        
        return ans

    def compress(self, rec_imgs: List[Image], org_img_i: Image, *args, **kwargs) -> List[Image]:
        torch.cuda.empty_cache()
        
        rec = rec_imgs[0]
        rec_upsample = rec_imgs[1]

        self.d_ver = self.get_owner_param('s_ver')
        self.d_hor = self.get_owner_param('s_hor')
        self.cal_parameter(self.d_ver, self.d_hor)
        beta = self.get_base_model_beta()
        img = org_img_i.clone()
        img.to_YUV_()
        img.convert_range_(rec.data_range)
        self.c = cat((img.get_component('b').clone(), img.get_component('c').clone()), dim=1)
        self.luma = self.downsample(img.get_component('a').clone())
        model_id = self.get_base_model_id()

        h, w = self.luma.shape[-2:]

        ans = rec.clone()
        self.NonlinearFilterWeights_U = []
        self.NonlinearFilterWeights_V = []
        self.NonLinearFilter_U_enabled = 1
        self.NonLinearFilter_V_enabled = 1
        
        before_u = 10*log10(255**2/mean((self.c[:,0:1] - ans.get_component('b'))**2))
        before_v = 10*log10(255**2/mean((self.c[:,1:2] - ans.get_component('c'))**2))

        self.LumaAidedAdaptiveNonlinearFilter_encoder(ans.clone())
        ans2 = self.LumaAidedAdaptiveNonlinearFilter_apply(ans.clone())
        ans = rec.clone()

        after_u = 10*log10(255**2/mean((self.c[:,0:1] - ans2.get_component('b'))**2))
        after_v = 10*log10(255**2/mean((self.c[:,1:2] - ans2.get_component('c'))**2))
        loss_u = beta * (after_u - before_u) - (len(self.NonlinearFilterWeights_U)*self.wP*self.lossModifier/4)/(h*w)
        loss_v = beta * (after_v - before_v) - (len(self.NonlinearFilterWeights_V)*self.wP*self.lossModifier/4)/(h*w)
        if loss_u < 0:
            self.NonLinearFilter_U_enabled = 0
            ans2.set_component('b',ans.get_component('b'))
        if loss_v < 0:
            self.NonLinearFilter_V_enabled = 0
            ans2.set_component('c',ans.get_component('c'))
        ans = ans2.clone()

        loss_best_u, loss_best_v = 0, 0
        mask_best_u, mask_best_v = zeros((1,1,0,0)),zeros((1,1,0,0))
        if rec_upsample is not None:
            bSizes = [128,112,96,80,64]
            for self.bSize in [bSizes[model_id]]:
                self.calculateOnOff(rec_upsample.clone(),ans.clone(),org_img_i=org_img_i.clone())
                ans2 = self.apply_OnoffSwitch(rec_upsample.clone(),ans.clone())
                h,w = ans.get_component('a').shape[-2:]
                before_u = 10*log10(255**2/mean((self.c[:,0:1] - ans.get_component('b'))**2))
                before_v = 10*log10(255**2/mean((self.c[:,1:2] - ans.get_component('c'))**2))
                after_u = 10*log10(255**2/mean((self.c[:,0:1] - ans2.get_component('b'))**2))
                after_v = 10*log10(255**2/mean((self.c[:,1:2] - ans2.get_component('c'))**2))
                loss_u = beta * (after_u - before_u) - (self.mask1.shape[2]*self.mask1.shape[3]*self.lossModifier/2)/(h*w)
                loss_v = beta * (after_v - before_v) - (self.mask2.shape[2]*self.mask2.shape[3]*self.lossModifier/2)/(h*w)
                if loss_u > loss_best_u:
                    loss_best_u = loss_u
                    mask_best_u = self.mask1.clone()
                if loss_v > loss_best_v:
                    loss_best_v = loss_v
                    mask_best_v = self.mask2.clone()
        self.mask1 = mask_best_u
        self.mask2 = mask_best_v
        if rec_upsample is not None:
            ans = self.apply_OnoffSwitch(rec_upsample.clone(),ans.clone())
        torch.cuda.empty_cache()
        return [ans] + rec_imgs[1:]

    @determinism_on_eval
    def decompress(self, imgs: List[Image], *args, **kwargs) -> List[Image]:
        img = imgs[0]
        img_alt = imgs[1]
        self.d_ver = self.get_owner_param('s_ver')
        self.d_hor = self.get_owner_param('s_hor')
        self.cal_parameter(self.d_ver, self.d_hor)
        img = self.LumaAidedAdaptiveNonlinearFilter_apply(img)
        if self.mask1.shape[2] > 0 or self.mask1.shape[3] > 0: 
            img = self.apply_OnoffSwitch(img_alt,img)
        ans = [img] + imgs[1:]
        return ans

    def LumaAidedAdaptiveNonlinearFilter_encoder(self,ans: Image):
        unfold = Unfold((1,1),padding=0)

        rec_Y = self.downsample(ans.get_component('a').clone())
        rec_U = ans.get_component('b').clone()
        rec_V = ans.get_component('c').clone()
        hh,ww = rec_Y.shape[-2:]

        cand = []
        count_w = (ww + self.NonlinearFilter_tile_width_base - 1) // self.NonlinearFilter_tile_width_base
        count_h = (hh + self.NonlinearFilter_tile_height_base - 1) // self.NonlinearFilter_tile_height_base
        self.NonlinearFilter_tile_width = ((ww + 64 * count_w - 1) // (64*count_w)) * 64
        self.NonlinearFilter_tile_height = ((hh + 64 * count_h - 1) // (64*count_h)) * 64
        for i in range ((hh+self.NonlinearFilter_tile_height - 1)//self.NonlinearFilter_tile_height):
            for j in range ((ww+self.NonlinearFilter_tile_width - 1)//self.NonlinearFilter_tile_width):
                h_start = float(i*self.NonlinearFilter_tile_height)/hh
                h_end = float(min((i+1)*self.NonlinearFilter_tile_height,hh))/hh
                v_start = float(j*self.NonlinearFilter_tile_width)/ww
                v_end = float(min((j+1)*self.NonlinearFilter_tile_width,ww))/ww
                cand.append([h_start,h_end,v_start,v_end])
        self.NonlinearFilterWeights_U = []
        self.NonlinearFilterWeights_V = []
        self.lumaMin = []
        self.lumaMax = []
        relu = ReLU()
        for c in cand:
            slice1 = slice(round(c[0]*hh), round(c[1]*hh))
            slice2 = slice(round(c[2]*ww), round(c[3]*ww))
            minimum = torch.round(torch.min(rec_Y[:,:,slice1,slice2])*100)
            self.lumaMin.append(minimum)
            maximum = torch.round(torch.max(rec_Y[:,:,slice1,slice2])*100)
            self.lumaMax.append(maximum)
            gap = ((maximum/100) - (minimum/100))/self.hist_split_num
            toCollect = []
            for i in range (self.hist_split_num):
                ceiling = (minimum/100) + i*gap
                toCollect.append(unfold(relu(rec_Y[:,:,slice1,slice2] - ceiling)).permute(0,2,1))
            A = cat(toCollect,dim=2)
            B1 = unfold((self.c[:,0:1] - rec_U)[:,:,slice1,slice2]).permute(0,2,1)
            B2 = unfold((self.c[:,1:2] - rec_V)[:,:,slice1,slice2]).permute(0,2,1)
            X1 = torch.linalg.lstsq(A, B1).solution.squeeze()
            X2 = torch.linalg.lstsq(A, B2).solution.squeeze()
            for x1,x2 in zip (X1,X2):
                self.NonlinearFilterWeights_U.append(self.integerize(x1))
                self.NonlinearFilterWeights_V.append(self.integerize(x2))
                    
    def LumaAidedAdaptiveNonlinearFilter_apply(self,ans):
        conv1 = Conv2d(2,8,1,bias = True,device = ans.get_component('a').device)
        relu = ReLU()
        conv2 = Conv2d(8,1,1,bias = False,device = ans.get_component('a').device)
        conv1.weight[:,:,:,:] = 0
        conv1.bias[:] = 0
        conv2.weight[:,:,:,:] = 0
        conv1.weight[0,1,0,0] = 1
        conv2.weight[0,0,0,0] = 1 
        conv1.weight[1:8,0,0,0] = 1 
        rec_Y = self.downsample(ans.get_component('a'))
        hh,ww = rec_Y.shape[-2:]
        for j in range(2):
            comp = 'b' if j ==0 else 'c'
            if not (self.NonLinearFilter_U_enabled and comp =='b' or self.NonLinearFilter_V_enabled and comp =='c'):
                continue
            weights = self.NonlinearFilterWeights_U if j ==0 else self.NonlinearFilterWeights_V
            if j == 0 and self.NonLinearFilter_U_enabled or j == 1 and self.NonLinearFilter_V_enabled:
                rec = ans.get_component(comp)
                input = cat((rec_Y,rec),dim=1)
                k = 0
                cand = []
                for i in range ((hh+self.NonlinearFilter_tile_height - 1)//self.NonlinearFilter_tile_height):
                    for j in range ((ww+self.NonlinearFilter_tile_width - 1)//self.NonlinearFilter_tile_width):
                        h_start = float(i*self.NonlinearFilter_tile_height)/hh
                        h_end = float(min((i+1)*self.NonlinearFilter_tile_height,hh))/hh
                        v_start = float(j*self.NonlinearFilter_tile_width)/ww
                        v_end = float(min((j+1)*self.NonlinearFilter_tile_width,ww))/ww
                        cand.append([h_start,h_end,v_start,v_end])
                weights_float = self.deinteger(weights).to(ans.get_component('a').device)
                for idx, c in enumerate(cand):
                    slice1 = slice(round(c[0]*hh), round(c[1]*hh))
                    slice2 = slice(round(c[2]*ww), round(c[3]*ww))
                    minimum = (float(self.lumaMin[idx])/100)
                    gap = ((float(self.lumaMax[idx])/100) - minimum)/self.hist_split_num
                    ceiling = (torch.ones((8),device=ans.get_component('a').device) * minimum + self.onetwothree.to(rec_Y.device)*gap)*(-1)
                    conv1.weight[0,0,0,0] = weights_float[k]
                    conv1.bias[0] = weights_float[k]*ceiling[0]
                    conv1.bias[1:8] = ceiling[1:8]
                    conv2.weight[0,1:8,0,0] = weights_float[k+1:k+8]
                    k += 8
                    rec[:,:,slice1,slice2]  = conv2(relu(conv1(input[:,:,slice1,slice2])))

        return ans

    def decode_header(self, ec: ECModule):
        self.minnn = int(ec.decode([1], max_symbol_value=2**self.wP-1, name='minSymbol'))
        self.maxSymbol = int(ec.decode([1], max_symbol_value=2**self.wP-1, name='maxSymbol'))

        mask_on_off_1 = int(ec.decode([1], max_symbol_value=1, name='mask1_enabled_flag'))
        mask_on_off_2 = int(ec.decode([1], max_symbol_value=1, name='mask2_enabled_flag'))
        h1,h2,w1,w2 = 0,0,0,0
        if mask_on_off_1 or mask_on_off_2:
            self.bSize = int(ec.decode([1], max_symbol_value=2**10-1, name='bS'))
            h = int(ec.decode([1], max_symbol_value=2**10-1, name='len_mask_y'))
            w = int(ec.decode([1], max_symbol_value=2**10-1, name='len_mask_x'))
            h1 = h if mask_on_off_1 else 0
            w1 = w if mask_on_off_1 else 0
            h2 = h if mask_on_off_2 else 0
            w2 = w if mask_on_off_2 else 0


        self.mask1 = ec.decode([1,1,h1,w1], max_symbol_value=2, name='W5[i,j,0]')
        self.mask2 = ec.decode([1,1,h2,w2], max_symbol_value=2, name='W5[i,j,1]')
        
        
        self.NonLinearFilter_U_enabled = int(ec.decode([1], max_symbol_value=1, name='nonLinear_enabled_U_flag'))
        self.logger.debug(f'self.NonLinearFilter_U_enabled: {self.NonLinearFilter_U_enabled}')
        self.NonLinearFilter_V_enabled = int(ec.decode([1], max_symbol_value=1, name='nonLinear_enabled_V_flag'))
        self.logger.debug(f'self.NonLinearFilter_V_enabled: {self.NonLinearFilter_V_enabled}')
        self.NonlinearFilterWeights_U, self.NonlinearFilterWeights_V = [], []
        if self.NonLinearFilter_V_enabled or self.NonLinearFilter_U_enabled:
            self.NonlinearFilter_tile_width = int(ec.decode([1], max_symbol_value=2**16-1, name='nonlinearW'))
            self.logger.debug(f'self.NonlinearFilter_tile_width: {self.NonlinearFilter_tile_width}')
            self.NonlinearFilter_tile_height = int(ec.decode([1], max_symbol_value=2**16-1, name='nonlinearH'))
            self.logger.debug(f'self.NonlinearFilter_tile_height: {self.NonlinearFilter_tile_height}')
            length = int(ec.decode([1], max_symbol_value=2**16-1, name='numTiles'))
            self.lumaMin = (ec.decode([length], max_symbol_value=2**16-1, name='minLuma'))
            self.logger.debug(f'lumaMin: {self.lumaMin}')
            self.lumaMax = (ec.decode([length], max_symbol_value=2**16-1, name='maxLuma'))
            self.logger.debug(f'lumaMax: {self.lumaMax}')


            if self.NonLinearFilter_U_enabled:
                cand_num_U = int(ec.decode([1], max_symbol_value=2**16-1, name='candNum'))
                self.logger.debug(f'NonLinearFilter weight num U: {cand_num_U}')
                self.NonlinearFilterWeights_U = ec.decode([cand_num_U], max_symbol_value=self.maxSymbol, name='A1')+self.minnn
            if self.NonLinearFilter_V_enabled:
                cand_num_V = int(ec.decode([1], max_symbol_value=2**16-1, name='candNum'))
                self.logger.debug(f'NonLinearFilter weight num V: {cand_num_V}')
                self.NonlinearFilterWeights_V = ec.decode([cand_num_V], max_symbol_value=self.maxSymbol, name='A1')+self.minnn
           



    def encode_header(self, ec: ECModule):
        self.maxxx = int(2**self.wP - 1)
        self.minnn = int(0)

        if self.NonLinearFilter_U_enabled:
            for i, add in enumerate(self.NonlinearFilterWeights_U):
                    self.minnn = min(add,self.minnn)
                    self.maxxx = max(add,self.maxxx)
        if self.NonLinearFilter_V_enabled:
            for i, add in enumerate(self.NonlinearFilterWeights_V):
                    self.minnn = min(add,self.minnn)
                    self.maxxx = max(add,self.maxxx)
        self.maxSymbol = self.maxxx - self.minnn
        ec.encode(self.minnn, max_symbol_value=2**self.wP-1, name='minSymbol')
        ec.encode(self.maxSymbol, max_symbol_value=2**self.wP-1, name='maxSymbol')

        ec.encode(int(self.mask1.shape[2]>0), max_symbol_value=1, name='mask1_enabled_flag')
        ec.encode(int(self.mask2.shape[2]>0), max_symbol_value=1, name='mask2_enabled_flag')
        mask_h = max(self.mask1.shape[2], self.mask2.shape[2])
        mask_w = max(self.mask1.shape[3], self.mask2.shape[3])
        if mask_h > 0:
            ec.encode(int(self.bSize), max_symbol_value=2**10-1, name='bS')
            ec.encode(int(mask_h), max_symbol_value=2**10-1, name='len_mask_y')
            ec.encode(int(mask_w), max_symbol_value=2**10-1, name='len_mask_x')
        
        ec.encode(self.mask1, max_symbol_value=2, name='W5[i,j,0]')
        ec.encode(self.mask2, max_symbol_value=2, name='W5[i,j,1]')


        ec.encode(int((self.NonLinearFilter_U_enabled)), max_symbol_value=1, name='nonLinear_enabled_U_flag')
        self.logger.debug(f'self.NonLinearFilter_U_enabled {self.NonLinearFilter_U_enabled}')
        ec.encode(int((self.NonLinearFilter_V_enabled)), max_symbol_value=1, name='nonLinear_enabled_V_flag')
        self.logger.debug(f'self.NonLinearFilter_V_enabled {self.NonLinearFilter_V_enabled}')

        if self.NonLinearFilter_V_enabled or self.NonLinearFilter_U_enabled:
            ec.encode(int((self.NonlinearFilter_tile_width)), max_symbol_value=2**16-1, name='nonlinearW')
            self.logger.debug(f'self.NonlinearFilter_tile_width {self.NonlinearFilter_tile_width}')
            ec.encode(int((self.NonlinearFilter_tile_height)), max_symbol_value=2**16-1, name='nonlinearH')
            self.logger.debug(f'self.NonlinearFilter_tile_height {self.NonlinearFilter_tile_height}')
            ec.encode(len(self.lumaMin),max_symbol_value = 2**16-1, name='numTiles')
            ec.encode(self.lumaMin, max_symbol_value=2**16-1, name='minLuma')
            self.logger.debug(f'self.LumaMin {[x.item() for x in self.lumaMin]}')
            ec.encode(self.lumaMax, max_symbol_value=2**16-1, name='maxLuma')
            self.logger.debug(f'self.lumaMax {[x.item() for x in self.lumaMax]}')

            if self.NonLinearFilter_U_enabled:
                ec.encode(int((len(self.NonlinearFilterWeights_U))), max_symbol_value=2**16-1, name='candNum')
                self.logger.debug(f'NonLinearFilter weight num U {len(self.NonlinearFilterWeights_U)}')
                for i, add in enumerate(self.NonlinearFilterWeights_U):
                    ec.encode(int(add - self.minnn), max_symbol_value=self.maxSymbol, name='A1')
        
            if self.NonLinearFilter_V_enabled:
                ec.encode(int((len(self.NonlinearFilterWeights_V))), max_symbol_value=2**16-1, name='candNum')
                self.logger.debug(f'NonLinearFilter weight num V {len(self.NonlinearFilterWeights_V)}')
                for i, add in enumerate(self.NonlinearFilterWeights_V):
                    ec.encode(int(add - self.minnn), max_symbol_value=self.maxSymbol, name='A1')
    

    