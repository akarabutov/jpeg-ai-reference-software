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
import torch.nn.functional as F
from src.codec.coding_tools.interfaces import CoderEngine
import einops
from .params import SkipModeParams
from src.codec.entropy_coding import HeaderCoder
from src.codec.components.contexts.utils import ContextUtils
from src.codec.common import tiling

class Data:
    skip_mode: bool = True
    cube_flags: torch.Tensor = None
    cube_flags_full: torch.Tensor = None

class IterObject:
    def __init__(self, stage: int, obj: "SkipModeCoder", y_hat_orig: torch.Tensor, mask: torch.Tensor, data: Data):
        self.obj = obj
        self.stage = stage
        self.y_hat_orig = y_hat_orig
        self.data = data
        self.mask_arr = mask
        
    def mask(self):
        ans = self.mask_arr
        if self.data.cube_flags_full is not None:
            ans = torch.logical_or(ans, self.data.cube_flags_full)
        return ans
    
    def check(self, y_hat_rec: torch.Tensor):
        if self.stage == 0:
            self.data.skip_mode = 1
            self.data.cube_flags = self.obj.gen_skip_cubeflag(y_hat_rec, self.y_hat_orig)
            self.data.cube_flags_full = self.obj.convert_cubeflag_map(self.data.cube_flags, self.y_hat_orig.shape)
        elif self.stage == 1:
            self.data.skip_mode = self.obj.check_skipmode(y_hat_rec, self.y_hat_orig)
            if not self.data.skip_mode:
                self.data.cube_flags = torch.zeros_like(self.data.cube_flags)
                self.data.cube_flags_full = self.obj.convert_cubeflag_map(self.data.cube_flags, self.y_hat_orig.shape)


class SkipModeCoder(CoderEngine):
    def __init__(self, chs_ls: int, **kwargs): 
        super(SkipModeCoder, self).__init__(use_coding_headers=True, has_enabled_flag=True, signal_enabled_flag=False, enabled=True, **kwargs)
        self._params_skip_mode = SkipModeParams()
        self.cube_flag = None
        self.cube_flags_full = None
        self.chs_ls = chs_ls
        self.cube_chan = chs_ls
        self.cube_size = 8   
        self.cube_group_size = 8  
        self.cur_skip_mask = None
        self.block_size = None
        self.output_shape = None
        self.skip_thr_precision = 7
        self.skip_block_size = 1
        self.thr_skip = 382
        self.skip_enable_flag = 1
        self.mV = {
            'skipFlag': 1,
            'blockSize': 255,
        }
        
    def _params_loaded(self):
        """Hook which executes when all parameters were loaded
        """
        self.params.load_params_from_owner(self._params_skip_mode.get_params_name_list())
        super()._params_loaded()

    def _nchw_to_nhwc(self, input_tensor):
        if input_tensor is not None:
            output_tensor = input_tensor.permute(0, 2, 3, 1).contiguous()
        else:
            output_tensor = input_tensor
        return output_tensor

    def _nhwc_to_nchw(self, input_tensor):
        if input_tensor is not None:
            output_tensor = input_tensor.permute(0, 3, 1, 2).contiguous()
        else:
            output_tensor = input_tensor
        return output_tensor
   
    def convert_cubeflag_map(self, init_map: torch.Tensor, output_shape: torch.Size) -> torch.Tensor:
        cube_flags_full = einops.repeat(init_map, 'a b c d -> a (b repeat1) (c repeat2) (d repeat3)', repeat1=self.cube_chan, repeat2=self.cube_size, repeat3=self.cube_size)            
        part0, part1, part2, part3 = torch.chunk(cube_flags_full, 4, dim=1)
        cube_flags_full = ContextUtils.up_shuffle((part0, part3, part1, part2))
        _,y_chan, y_hei, y_wid = output_shape
        cube_flags_full = ~ cube_flags_full[:, :y_chan, :y_hei, :y_wid]        
        return cube_flags_full
        
        
    def gen_skip_cubeflag(self, y_hat: torch.Tensor, y_org: torch.Tensor) -> torch.Tensor:
        #convert to 4C,h5,w5 space to align with Y componennt of MCM
        height, width = y_hat.shape[-2:]
        y_hat_padding = F.pad(y_hat, (0, width % 2, 0, height % 2))
        y_hat = ContextUtils.down_shuffle(y_hat_padding)
        part0, part3, part1, part2 = y_hat
        y_hat = torch.cat((part0, part1, part2, part3), dim=1)
        y_org_padding = F.pad(y_org, (0, width % 2, 0, height % 2))
        y_org = ContextUtils.down_shuffle(y_org_padding)
        part0, part3, part1, part2 = y_org
        y_org = torch.cat((part0, part1, part2, part3), dim=1)
        diff_yhat = torch.abs(y_hat - y_org)
        N,C,H,W = diff_yhat.shape
        diff_yhat = diff_yhat.reshape(1,N,C,H,W)
        cube_size = self.cube_size
        cube_chan = self.cube_chan
        h_pad = ((H+cube_size-1)// cube_size) * cube_size - H
        w_pad = ((W+cube_size-1)// cube_size) * cube_size - W
        diff_yhat = F.pad(diff_yhat, (0, w_pad, 0, h_pad), value=0)
        maxpool = torch.nn.MaxPool3d((cube_chan, cube_size, cube_size), (cube_chan, cube_size, cube_size), 0)
        cubeflag = (maxpool(diff_yhat) > self.skip_cube_thr) # skip_mask = cubeflag ! skip_masp
        cubeflag = ~cubeflag[0,:,:,:,:]  
        return cubeflag

        
    def check_skipmode(self, y_hat: torch.Tensor, y_org: torch.Tensor) -> bool:
        diff_yhat = torch.abs(y_hat - y_org)
        max_diff = diff_yhat.max()
        if max_diff > self.skip_judge_thr:
            return 0
        else:
            return 1      

    def encode_header(self, ec: HeaderCoder):
        ccs_id = self.get_owner_param("ccs_id")
        use_cube_flags = 0 if self.cube_flag.all().item() else 1
        ec.encode(use_cube_flags, 1, name=f'use_cube_flags[{ccs_id}]')
        if use_cube_flags:
            cube_flag_num = self.cube_flag.numel()
            cube_flag_chanfirst = self.cube_flag #.permute(0,2,3,1)
            cube_flag_vector = cube_flag_chanfirst.reshape(cube_flag_num)
            group_num = (cube_flag_num + self.cube_group_size -1) // self.cube_group_size
            cube_flag_vector_pad = torch.ones(group_num*self.cube_group_size)
            cube_flag_vector_pad[0:cube_flag_num] = cube_flag_vector.int()
            cube_group_flag = cube_flag_vector_pad.reshape(group_num, -1).min(1)[0]
            cube_group_flag = 1 - cube_group_flag
            self.logger.debug(f"Encode cube flags: {self.cube_flag.sum() * 100.0/ cube_flag_num}%")
            
            for i in range(group_num-1):
                ec.encode(cube_group_flag[i].int(), 1, name=f"cube_group_flag[{ccs_id}]")
                if cube_group_flag[i] > 0:
                    ec.encode(cube_flag_vector[i*self.cube_group_size: (i+1)*self.cube_group_size].int(), 1, name=f"cube_flag[{ccs_id}]")
            
            ec.encode(cube_group_flag[group_num-1].int(), 1, name=f"cube_group_flag[{ccs_id}]")
            if cube_group_flag[group_num-1] > 0:
                ec.encode(cube_flag_vector[(group_num-1)*self.cube_group_size: cube_flag_num].int(), 1, name=f"cube_flag[{ccs_id}]")
    
    def decode_header(self, ec: HeaderCoder):
        self.skip_block_size = 1
        self.thr_skip = 382
        ccs_id = self.get_owner_param("ccs_id")
        use_cube_flags = ec.decode(1, 1, name=f'use_cube_flags[{ccs_id}]').item()
        if use_cube_flags:
            h_ls, w_ls = self.get_ls_shape(0)
            self.cube_size = 8 
            self.cube_chan = self.chs_ls 
            cubeH = ((h_ls + 1) // 2 + self.cube_size-1) // self.cube_size
            cubeW = ((w_ls + 1) // 2 + self.cube_size-1) // self.cube_size
            cubeC = 4 * self.chs_ls // self.cube_chan
            cube_flag_num = cubeC*cubeH*cubeW
            group_num = (cube_flag_num + self.cube_group_size -1)// self.cube_group_size
            cube_flag = torch.ones(cube_flag_num)
            for i in range(group_num-1):
                value = ec.decode(1, 1, name=f"cube_group_flag[{ccs_id}]")
                if value > 0:
                    cube_flag_sub = ec.decode(self.cube_group_size, 1, name=f"cube_flag[{ccs_id}]")
                    cube_flag[i*self.cube_group_size: (i+1)*self.cube_group_size] = cube_flag_sub
            
            value = ec.decode(1, 1, name=f"cube_group_flag[{ccs_id}]")
            if value > 0:
                cube_flag_sub = ec.decode(cube_flag_num - (group_num-1)*self.cube_group_size, 1, name=f"cube_flag[{ccs_id}]")
                cube_flag[(group_num-1)*self.cube_group_size: cube_flag_num] = cube_flag_sub
            
            #cube_flag = cube_flag.to(device=next(self.hyper_decoder.parameters()).device)
            cube_flag = cube_flag > 0
            '''
            cube_flag = cube_flag.view(1, cubeH, cubeW, cubeC)
            self.cube_flag = cube_flag.permute(0, 3, 1, 2)
            '''
            cube_flag = cube_flag.view(1, cubeC, cubeH, cubeW)
            self.cube_flag = cube_flag
            self.cube_flags_full = self.convert_cubeflag_map(cube_flag, (1, self.chs_ls, h_ls, w_ls))
            self.logger.debug(f"Decode cube flags: {cube_flag.sum() * 100.0/ cube_flag.numel()}%")
        else:
            self.cube_flag = None
            self.cube_flags_full = None
       
    def generate_skip_mask(self, scale_hat):
        block_size = self.skip_block_size
        
        _, _, h, w = scale_hat.shape
        h_pad = ((h + block_size - 1) // block_size) * block_size - h
        w_pad = ((w + block_size - 1) // block_size) * block_size - w

        likely = F.pad(scale_hat.float(), (0, w_pad, 0, h_pad), value=2**self.unscaled_sigma_precision).int()
        
        denom = self.skip_thr_precision
        power = self.unscaled_sigma_precision - denom

        if power > 0:
            thr_min = self.thr_skip*pow(2,abs(power))
        else:
            thr_min = self.thr_skip
            likely = likely*pow(2,abs(power))
        
        mask = (likely > thr_min)
        if block_size > 1:
            mask = einops.repeat(mask,
                                 'a b c d -> a b (c repeat1) (d repeat2)',
                                 repeat1=block_size,
                                 repeat2=block_size)
        mask = F.pad(mask, (0, -w_pad, 0, -h_pad), value=1)
        return mask, block_size 
           
    @staticmethod
    def mask_arr(arr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return arr
        else:
            return arr[mask]
        
    @staticmethod
    def unmask_arr(arr: torch.Tensor, mask:torch.Tensor, output_shape: torch.Size) -> torch.Tensor:
        if mask is None:
            return arr.view(output_shape)
        else:
            ans = torch.zeros(output_shape, device=arr.device, dtype=arr.dtype)
            ans[mask] = arr
            return ans
        
    def mask(self):
        if self.skip_enable_flag == 0 or not self.enabled:
            return None
        ans = self.cur_skip_mask
        if self.cube_flags_full is not None:
            if ans.device != self.cube_flags_full.device:
                self.cube_flags_full = self.cube_flags_full.to(ans.device)
            ans = torch.logical_or(ans, self.cube_flags_full)        
        return ans
    
    def _update_mask(self, scale_log: torch.Tensor):
        if scale_log is not None:
            self.cur_skip_mask, _ = self.generate_skip_mask(scale_log)
            self.output_shape = scale_log.shape
    
    def analyze():
        pass

    def mask_with_tile(self, tile: tiling.Area):
        if self.skip_enable_flag == 0 or not self.enabled:
            return None
        mask_tile =  tiling.get_data(self.cur_skip_mask, tile)
        if self.cube_flags_full is not None:
            if mask_tile.device != self.cube_flags_full.device:
                self.cube_flags_full = self.cube_flags_full.to(mask_tile.device)
            cube_flags_full_tile =  tiling.get_data(self.cube_flags_full, tile)
            mask_tile = torch.logical_or(mask_tile, cube_flags_full_tile)
        return mask_tile, mask_tile.shape
