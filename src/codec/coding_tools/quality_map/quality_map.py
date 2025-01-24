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

import math
import os
import bisect
from time import time
from typing import Dict, List, Tuple
import torch
import numpy as np
import torch.nn.functional as F
import numpy as np

from src.codec.entropy_coding import HeaderCoder,ECModule,ECSettingContext
from src.codec.common import Decisions
from .params import QualityMapParams
from src.codec.common import TensorOps
from src.codec.coding_tools.interfaces import CoderEngine
from src.codec.coding_tools.quantization.params import QuantizerParams


class QualityMap(CoderEngine):
    def __init__(self,  *args, **kwargs):
        super(QualityMap, self).__init__(has_enabled_flag=True, enabled=0, *args, **kwargs)
        self._params_qual_map = QualityMapParams()
        self.quantizer_params = QuantizerParams()

        self.qp_ec_index = 0
        self.sigma_list = [0, 4, 5, 6, 8, 10, 14, 18] 
        self.device = 'cuda'
        
        self.q_scale_log_map = [-886, -775, -664, -554, -443, -332, -221, -111, 0, 111, 221, 332, 443, 554, 664, 775, 886]
        self.q_scale_map = [4,5,6,7,8,10,12,14,16,20,23,27,32,39,46,54,64]
        assert len(self.q_scale_log_map) == len(self.q_scale_map)
        
        #self.entropy = 
        
    def clip_qpmap(self, qp_map: torch.Tensor) -> torch.Tensor:
        min_v = -len(self.q_scale_log_map) // 2
        max_v = -min_v
        return qp_map.clamp(min_v, max_v)
        
    def _params_loaded(self) -> None:
        self.alignment_size = self.get_alignment_size()
        self.set_ec_params()

    def decode_header(self, ec: HeaderCoder) -> None:
        """Decode signalling type for tiling

        Args:
            ec (ECModule): entropy coder
        """
        # TODO: Add num_threads_q decoding!
        multi_threading_q = ec.decode([1], bits_count=1, name='multi_threading_q').item()
        if multi_threading_q:
            log2_num_threads_q_minus1 = ec.decode([1], bits_count=1, name='log2_num_threads_q_minus1').item()
            self.num_threads = 1 << (log2_num_threads_q_minus1+1)
            
        self.qp_ec_index = int(ec.decode([1], max_symbol_value=7, name='quality_map_entropy_index')).item()
        self.set_ec_params()

    def encode_header(self, ec: HeaderCoder) -> None:
        """Encode signalling type for tiling

        Args:
            ec (ECModule): entropy coder
        """
        multi_threading_q = self.num_threads > 1
        ec.encode(multi_threading_q, bits_count=1, name='multi_threading_q')
        if multi_threading_q:
            tmp = math.log2(self.num_threads)
            assert(abs(tmp-int(tmp)) < 1E-5)
            ec.encode(int(tmp)-1, bits_count=2, name='log2_num_threads_q_minus1')
        ec.encode(self.qp_ec_index, max_symbol_value=7, name='quality_map_entropy_index')

    def encode(self, ec: ECModule, decision: Decisions, *args, **kwargs) -> None:
        """generate bit stream for qp map

        Args:
            ec (ECModule): entropy model.
            decision (Decisions): results of qp map.
            h (int): height of the image.
            w (int): width of the image.
        """
        qp_map = decision['qp_map']
        delta_qp_map = torch.zeros_like(qp_map)
        h, w = qp_map.shape[-2:]
        for i in range(h):
            for j in range(w):
                if i == 0  and j > 0:
                    delta_qp_map[:,:,i,j] = qp_map[:,:,i,j] - qp_map[:,:,i,j-1]
                elif i > 0  and j == 0:
                    delta_qp_map[:,:,i,j] = qp_map[:,:,i,j] - qp_map[:,:,i-1,j]
                elif i > 0  and j > 0:
                    qp_pred = (qp_map[:,:,i,j-1] + qp_map[:,:,i-1,j])//2
                    delta_qp_map[:,:,i,j] = qp_map[:,:,i,j] - qp_pred
                else:
                    delta_qp_map[:,:,i,j] = qp_map[:,:,i,j]
        
        index = self.qp_ec_index
        
        sigma_list = self.sigma_list 
        # use list to get sigma value
        assert(index <= (len(sigma_list)-1))
        sigma = sigma_list[index]
        sigma = int(sigma * (2**self.sigma_precision))
        self.logger.debug(f'Encoding qp_map with one sigma:{index}')
        symbols = delta_qp_map[:]
        scale_hat = (torch.ones_like(symbols) * sigma).int()
        masks = torch.ones_like(scale_hat).bool()
        with self.set_ec_context(ec, "qmap"):
            ec.encode_sgt(symbols,
                        scale_hat,
                        masks,
                        entropy_prob_model=self.get_sgm_entropy_model(),
                        name=f'{self.name} qp_map')

    def set_ec_params(self) -> None:
        EC = self.get_object_by_url('ce.EC')
        setattr(EC, 'num_threads_q', self.num_threads)
        
 
    def decode(self, ec: ECModule, *args, **kwargs) -> Decisions:
        """generate bit stream for qp map

        Args:
            ec (ECModule): entropy model.
            decision (Decisions): results of qp map.
            h (int): height of the image.
            w (int): width of the image.
        """
        logger = self.logger

        h_pad2, w_pad2 = self.get_ls_shape(0)
        qp_map_shape = (1, 1, h_pad2, w_pad2)
        qp_map = torch.zeros(qp_map_shape)
        logger.debug('Decoding qp_map')
        index = self.qp_ec_index
        
        sigma_list = self.sigma_list
        # use list to get sigma value
        assert(index <= (len(sigma_list)-1))
        sigma = sigma_list[index]
        sigma = int(sigma * (2**self.sigma_precision))
        scale_hat = (torch.ones(qp_map_shape) * sigma).int()
        masks = torch.ones_like(scale_hat).bool()
        with self.set_ec_context(ec, "qmap"):
            delta_qp_map = ec.decode_sgt(scale_hat, masks,
                            entropy_prob_model=self.get_sgm_entropy_model(),
                            name=f'{self.name} qp_map')
        delta_qp_map = delta_qp_map.view(qp_map_shape)
        
        for i in range(h_pad2):
            for j in range(w_pad2):
                if i == 0  and j > 0:
                    qp_map[:,:,i,j] = delta_qp_map[:,:,i,j] + qp_map[:,:,i,j-1]
                elif i > 0  and j == 0:
                    qp_map[:,:,i,j] = delta_qp_map[:,:,i,j] + qp_map[:,:,i-1,j]
                elif i > 0  and j > 0:
                    qp_pred = (qp_map[:,:,i,j-1] + qp_map[:,:,i-1,j])//2
                    qp_map[:,:,i,j] = delta_qp_map[:,:,i,j] + qp_pred
                else:
                    qp_map[:,:,i,j] = delta_qp_map[:,:,i,j]

        ans = Decisions()
        ans['qp_map'] = qp_map
        return ans
    
    def analyze(self, decisions: Decisions = None) -> Decisions:
        if 'qp_map' in decisions:
            return decisions
        scale_log = decisions.get('scale_log', None)
        shape = scale_log.shape[-2:]
#                component, img_input_file, is_luma, **kwargs):
        qp_map_type = self.qp_map_type
        if qp_map_type == 1:
            args_roi_x_list = [int(x)//16 for x in self.roi_lt_pos_x_list.split(',')]
            args_roi_y_list = [int(x)//16 for x in self.roi_lt_pos_y_list.split(',')]
            args_roi_wid_list = [int(x)//16 for x in self.roi_wid_list.split(',')]
            args_roi_hei_list = [int(x)//16 for x in self.roi_hei_list.split(',')]
            qp_map = self.generate_qp_map_byROI(shape,args_roi_x_list,args_roi_y_list,args_roi_wid_list,args_roi_hei_list)
        elif qp_map_type == 2:
            qp_map = self.generate_qp_map_byNoise(shape)
        elif qp_map_type == 0:
            qp_map = self.generate_qp_map(shape)
        elif qp_map_type == 3:
            qp_map = self.generate_qp_map_byROI_map(self.adjust_qp, shape=shape) 
        ans = Decisions()
        ans['qp_map'] = qp_map
        return ans
    
    def matching_qp_to_scale_log(self, qp_map):
        q_scale_map = torch.zeros_like(qp_map)
        for i, v in enumerate(self.q_scale_log_map):
            q_scale_map[qp_map==(i-8)] = v
        return q_scale_map.int()
    
    def matching_qp_to_scale(self, qp_map):
        q_scale_map = torch.zeros_like(qp_map)
        for i, v in enumerate(self.q_scale_map):
            q_scale_map[qp_map==(i-8)] = v
        return q_scale_map.int()
    
    def quantize_resi(self, x: torch.Tensor, decisions: Decisions = None) -> torch.Tensor:
        if decisions is not None and 'qp_map' in decisions:
            scale_map_linear = self.matching_qp_to_scale(decisions.get('qp_map')).to(x.device)
            scaler_vec = scale_map_linear        
            return x * scaler_vec / 16
        else:
            return x
    
    def quantize_scale(self, x: torch.Tensor, decisions: Decisions = None) -> torch.Tensor:
        if decisions is not None and 'qp_map' in decisions:
            scale_map_log = self.matching_qp_to_scale_log(decisions.get('qp_map')).to(x.device)
            return x + scale_map_log
        else:
            return x
    
    def dequantize_resi(self, x: torch.Tensor, decisions: Decisions = None) -> torch.Tensor:
        if decisions is not None and 'qp_map' in decisions:
            scale_map_linear = self.matching_qp_to_scale(decisions.get('qp_map')).to(x.device)
            q_scale_map = (scale_map_linear)/16
            return x/(q_scale_map + 1e-9)
        else:
            return x
    
    def set_qp_ec_index(self, value):
        self.qp_ec_index = value

    def cal_var(self, img, window):
        N, _, iH, iW = img.shape
        var_h = (iH + window-1)//window
        var_w = (iW + window-1)//window
        var_map = torch.zeros((N,1,var_h,var_w))
        for i in range(var_h):
            for j in range(var_w):
                h_end = np.clip(window*(i+1), 0, iH)
                w_end = np.clip(window*(j+1), 0, iW)
                var_map[:,0,i,j] = torch.var(img[:, 0, window*i:h_end, window*j:w_end].reshape(N,-1), dim=1)
        
        return var_map
 
    def generate_qp_map_byROI(self, shape,args_roi_x_list,args_roi_y_list,args_roi_wid_list,args_roi_hei_list):
        h,w = shape
        roi_num = len(args_roi_x_list)
        down_size = 	self.alignment_size
        h_diff = int(math.ceil(h / down_size) * down_size) //down_size
        w_diff = int(math.ceil(w / down_size) * down_size) //down_size
        qp_map = -3*torch.ones((1,1,h_diff, w_diff))
        for i in range(roi_num):
            x = args_roi_x_list[i]
            y = args_roi_y_list[i]
            wid = args_roi_wid_list[i]
            hei = args_roi_hei_list[i]
            qp_map[0,0,y:y+hei, x:x+wid] = 0
        qp_map = self.clip_qpmap(qp_map)
        self.generate_qp_map_index(qp_map)
        return qp_map

    def generate_qp_map_byROI_map(self, adjust_qp, shape):
        #roi_name = os.path.basename(img_name).replace('.png', '_mask.png')
        # path_name = img_name.rsplit('/',1)[0] + '/mask'
        import cv2
        img_roi = cv2.imread(self.ROI_map_in_file)
        
        roi_mask_tensor = torch.tensor(img_roi.astype(dtype=np.int32), dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
        
        down_scale_x = math.ceil(roi_mask_tensor.shape[-1] / shape[-1])
        down_scale_y = math.ceil(roi_mask_tensor.shape[-2] / shape[-2])
        # h_diff = (down_scale_y * shape[-1] - roi_mask_tensor.shape[-1]) // 2
        # w_diff = (down_scale_x * shape[-2] - roi_mask_tensor.shape[-2]) // 2
        
        # keep maxpooling --> correct mask area && interpolation --> matched size with scale_log
        roi_mask_tensor = F.interpolate(roi_mask_tensor, size=[down_scale_y*shape[-2],down_scale_x*shape[-1]])
         
        roi_mask_tensor = F.max_pool2d(roi_mask_tensor, (down_scale_y, down_scale_x), (down_scale_y, down_scale_x))
        # roi_mask_tensor = F.interpolate(roi_mask_tensor, size=shape[-2:], mode="nearest")
        
        roi_mask = (roi_mask_tensor[:,0:1,:,:]==255)&(roi_mask_tensor[:,1:2,:,:]==255)&(roi_mask_tensor[:,2:3,:,:]==255)
        roi_mask = roi_mask.float()
        qp_map = torch.zeros_like(roi_mask)
        if adjust_qp:
            qp_map = -3*(1 - roi_mask) + 6*roi_mask
        
        qp_map = self.clip_qpmap(qp_map)
        '''
        roi_mask_np = roi_mask.cpu().numpy().astype(dtype=np.uint8)
        q_map_name =self.ROI_map_out_file + '_400_8bit.yuv'
        os.makedirs(os.path.dirname(self.ROI_map_out_file), exist_ok=True)
        f = open(q_map_name, 'wb')
        f.write(roi_mask_np*255)
        f.close()

        q_map_name = self.ROI_map_out_file + '_400_8bit.png'
        
        cv2.imwrite(q_map_name, roi_mask_np[0,0,:,:]*255)
        '''

        self.generate_qp_map_index(qp_map)
        return qp_map

    def generate_qp_map_byNoise(self, shape):
        h,w = shape
        down_size = 	self.alignment_size
        h_diff = int(math.ceil(h / down_size) * down_size) //down_size
        w_diff = int(math.ceil(w / down_size) * down_size) //down_size
        #qp_map = torch.randint(-8,8,(1,1,h_diff, w_diff))
        qp_map = torch.zeros((1,1,h_diff, w_diff))
        block_test = self.block_qp
        #import pdb; pdb.set_trace()
        for i in range(math.ceil(h_diff/block_test)):
            for j in range(math.ceil(w_diff/block_test)):
                    x_min = i*block_test
                    x_max = np.clip(i*block_test+block_test, 0, h_diff)
                    y_min = j*block_test
                    y_max = np.clip(j*block_test+block_test, 0, w_diff)
                    qp_map[0,0,x_min:x_max, y_min:y_max] = self.qp_max if (i+j)%2 == 1 else self.qp_min
        
        qp_map = self.clip_qpmap(qp_map)
        
        self.generate_qp_map_index(qp_map)
        return qp_map
 
    def generate_qp_map_zero(self, shape):
        h,w = shape
        down_size = 	self.alignment_size
        h_diff = int(math.ceil(h / down_size) * down_size) //down_size
        w_diff = int(math.ceil(w / down_size) * down_size) //down_size
        #qp_map = torch.randint(-8,8,(1,1,h_diff, w_diff))
        qp_map = torch.zeros((1,1,h_diff, w_diff))
        
        self.generate_qp_map_index(qp_map)
        return qp_map

    def generate_qp_map(self, shape):
        h,w = shape
        down_size = 	self.alignment_size
        h_diff = int(math.ceil(h / down_size) * down_size)  - h
        w_diff = int(math.ceil(w / down_size) * down_size)  - w
        img = torch.nn.functional.pad(img, (0, w_diff, 0, h_diff), mode='replicate')
        # 8X8 size
        #var_map = self.cal_var(img, 8)
        #maxpool = torch.nn.MaxPool2d((2, 2), (2, 2), 0)
        down_factor = self.alignment_size//4 #np.sqrt(self.alignment_size).round().astype(int)
        var_map = self.cal_var(img, 4)
        maxpool = torch.nn.MaxPool2d((down_factor, down_factor), (down_factor, down_factor), 0)
        var_map = -(maxpool(-var_map)) #cal mininual val
        var_mean = var_map.mean()
        qp_map = torch.zeros_like(var_map)
        qp_map += -1*((var_map > 0.8*var_mean).float())
        qp_map += -1*((var_map > 1.5*var_mean).float())
        qp_map += -1*((var_map > 2.5*var_mean).float())
        qp_map += -1*((var_map > 4*var_mean).float())
            #qp_map = -1*torch.clip((var_map-7)//2, 0, 4) # -1 means bit reducing 20%, -2 reducing 40%; -4 reducing 80%
        
        qp_map = self.clip_qpmap(qp_map)
        #q_map_name = './qp_map' + str(h) +'.pth'
        #torch.save(qp_map, q_map_name)
        
        self.generate_qp_map_index(qp_map)
        return qp_map

    def generate_qp_map_index(self, qp_map):
        """generate bit stream for qp map

        Args:
            ec (ECModule): entropy model.
            decision (Decisions): results of qp map.
            h (int): height of the image.
            w (int): width of the image.
        """
        delta_qp_map = torch.zeros_like(qp_map)
        _, _, h, w = qp_map.shape
        for i in range(h):
            for j in range(w):
                if i == 0  and j > 0:
                    delta_qp_map[:,:,i,j] = qp_map[:,:,i,j] - qp_map[:,:,i,j-1]
                elif i > 0  and j == 0:
                    delta_qp_map[:,:,i,j] = qp_map[:,:,i,j] - qp_map[:,:,i-1,j]
                elif i > 0  and j > 0:
                    qp_pred = (qp_map[:,:,i,j-1] + qp_map[:,:,i-1,j])//2
                    delta_qp_map[:,:,i,j] = qp_map[:,:,i,j] - qp_pred
        num_zero = (delta_qp_map == 0).sum() * 1.0 / (h*w)
        
        ## use bisect to get the index
        num_zero_list = [0.25, 0.5, 0.70, 0.80, 0.90, 0.95, 0.99]
        index = len(num_zero_list) - bisect.bisect(num_zero_list, num_zero)
            
        self.qp_ec_index = index

    
