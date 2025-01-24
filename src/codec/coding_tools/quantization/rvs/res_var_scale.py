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

from copy import deepcopy

import einops
import torch
import torch.nn.functional as F
from functools import wraps
from math import log2
import time
from src.codec.entropy_coding import HeaderCoder
from src.codec.common import Decisions, Log2LinConvertion
from ..base import QuantizationInterface
##
from .params import ResScaleParamsParams
import numpy as np
import os


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'----------------------->Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


class ResVarScale(QuantizationInterface):
    tiny_val = 1e-9
    """Class for handeling information about max symbol of y_hat
    """
    
    #@timeit
    def __init__(self, chs, log_k=(np.log(54.85)-np.log(0.11))/31, **kwargs):
        super(ResVarScale, self).__init__(has_enabled_flag=False, enabled=1, **kwargs)
        self.chs = chs
        self._params_rsm = ResScaleParamsParams()
        self.rvs_thr_precision = 13
        self.pad_val = 1411
        self.cwgf = []
        
        if not hasattr( self, 'RVSTables'):
            self.RVSTables = {"data":[0], "isBuilt":False}
            self.CD_text = [0]
            cwg_on_off = [0,1,2]
            rvs_enabled = [0,1]
            self.RVSTables["data"] = [0]*len(cwg_on_off)
            self.CD_text = [0]*len(cwg_on_off)
            for cwg in cwg_on_off:
                self.RVSTables["data"][cwg] = [0]*len(rvs_enabled)
                self.CD_text[cwg] = [0]*len(rvs_enabled)
                for rvs in rvs_enabled:
                    self.RVSTables["data"][cwg][rvs] = [0]
                    self.CD_text[cwg][rvs] = [0]
                    self.RVSTables["data"][cwg][rvs][0] = [0]*3
                    self.CD_text[cwg][rvs][0] = [0]*4
        self.log_k = log_k
        self.scaler1_log_cwg = torch.ones(chs) * 117
        self.scaler2_log_cwg = torch.ones(chs) * -33
        
    def export_models(self, output_dir: str, opset_version: int):
        return
    # Almost impossible to match these tables with variables in spec.
        self.buildTables()
        data = self.RVSTables["data"]
        for i in range(len(data)):
            for j in range(len(data[i])):
                for k in range(len(data[i][j])):
                    for l in range(len(data[i][j][k])):
                        if isinstance(data[i][j][k][l], torch.nn.Embedding):
                            data[i][j][k][l] = data[i][j][k][l].weight.view(-1).cpu().numpy()
        np_arr = np.array(data)
        os.makedirs(output_dir, exist_ok=True)
        for lst_id, lst_name in enumerate(['TableLin', 'TableLog', 'TableInvLin']):
            s_len = np_arr.shape[-1]
            arr = np.zeros((4,s_len), dtype=np_arr.dtype)
            for cwg in range(2):
                for rvs in range(2):
                    idx1 = rvs * 2 + cwg
                    assert not (np_arr[cwg][rvs][0].astype(np.int) - np_arr[cwg][rvs][0]).any()
                    arr[idx1,:] =  np_arr[cwg][rvs][0].astype(np.int)
                    np.savetxt(os.path.join(output_dir, f"{lst_name}.csv"), arr.astype(np.int), fmt='%d', delimiter=',')
       
    @property
    def threshold_rvs(self):
        return self.threshold_01
    
    @property
    def rvs_scale_list(self):
        return self.rvs_scale_list_id1
    
    def decode_header(self, ec: HeaderCoder) -> None:
        """Decode y_max_symbol from header of a bitstream

        Args:
            ec (HeaderCoder): entropy coder
        """
        ccs_id = self.get_owner_param('ccs_id')
        self.rvs_enabled = ec.decode([1], max_symbol_value=1, name=f'rvs_enable_flag[{ccs_id}]')
        self.cwg_enabled = ec.decode([1], max_symbol_value=1, name=f'grfs_enable_flag[{ccs_id}]')
        if self.cwg_enabled:
            num_chs = self.get_owner_param('num_chs', self.chs)
            self.cwgf = torch.zeros([self.chs, 1, 1], dtype=torch.int32)
            self.cwgf[:num_chs] = ec.decode(num_chs, 1, name=f'grfs_channel_flag[{ccs_id}]').unsqueeze(-1).unsqueeze(-1)
            
    def encode_header(self, ec: HeaderCoder) -> None:
        """Encode y_max_symbol to header of a bitstream

        Args:
            ec (HeaderCoder):  entropy coder
        """
        ccs_id = self.get_owner_param('ccs_id')
        ec.encode(self.rvs_enabled,
                        max_symbol_value=1,
                        name=f'rvs_enable_flag[{ccs_id}]')
        ec.encode(self.cwg_enabled,
                max_symbol_value=1,
                name=f'grfs_enable_flag[{ccs_id}]')  
        if self.cwg_enabled:
            num_chs = self.get_owner_param('num_chs', self.chs)
            ec.encode(self.cwgf[:num_chs], 1, name=f'grfs_channel_flag[{ccs_id}]')
    
    #@timeit
    def buildTables(self):

        if (not hasattr(self, 'RVSTables')) or (not self.RVSTables["isBuilt"]):
            
            self.table1 = torch.zeros_like(Log2LinConvertion.table)
            for i in range(len(Log2LinConvertion.table)):
                self.table1[i] = i
            
            cwg_on_off = [0,1,2]
            rvs_enabled = [0,1]
            for cwg in cwg_on_off:
                for rvs in rvs_enabled:
                    table2 = torch.zeros_like(Log2LinConvertion.table)
                    table3 = torch.zeros_like(Log2LinConvertion.table)
                    table4 = torch.zeros_like(Log2LinConvertion.table)
                    numFilters = 3 
                    thrList = deepcopy(self.threshold_rvs_id1)
                    
                    scale_list = deepcopy(self.rvs_scale_list_id1)
                    if rvs == 0:
                        thrList = []
                        scale_list = [128]
                        numFilters = 0
                    thrList.append(10000000)

                    for i in range(len(scale_list)):
                        if cwg == 1:
                            scale_list[i] = int(round(scale_list[i]*1.06))
                        if cwg == 2:
                            scale_list[i] = int(round(scale_list[i]*0.98))
                    denom = self.rvs_thr_precision
                    power = self.scaled_sigma_precision - denom
                    thr_min_log = -1000000
                    prev = [0,0,0,0] 
                    for i in range(numFilters+1):
                        thr_max = thrList[i]
                        thr_max = int(thr_max * (2**power))
                        thr_max_log = Log2LinConvertion.idx2log_table(torch.tensor(thr_max))
                        mask = torch.relu(self.table1-thr_min_log) - torch.relu(self.table1-thr_min_log - 1)
                        table2 += torch.round(mask.float() * (scale_list[i]-prev[0]))
                        
                        a = round(np.log (scale_list[i]/128)/ ((np.log(54.82)- np.log(0.11))/31/2**7))
                        table3 += mask * torch.tensor(a - prev[1])
                        b = round((2**7 * 2**16)/float(scale_list[i]))
                        table4 += mask * (b-prev[2])
                        thr_min_log = thr_max_log
                        if i == 0:
                            self.CD_text[cwg][rvs][0][0]=([scale_list[i]])
                            self.CD_text[cwg][rvs][0][1]=([a])
                            self.CD_text[cwg][rvs][0][2]=([round((2**7 * 2**16)/float(scale_list[i]))])
                            self.CD_text[cwg][rvs][0][3]=([ thr_max_log.item()])
                            prev = [scale_list[i],a,round((2**7 * 2**16)/float(scale_list[i])),thr_max_log.item()]
                        else:
                            self.CD_text[cwg][rvs][0][0].append(scale_list[i]-prev[0])
                            self.CD_text[cwg][rvs][0][1].append(a-prev[1])
                            self.CD_text[cwg][rvs][0][2].append(round((2**7 * 2**16)/float(scale_list[i]))-prev[2])
                            self.CD_text[cwg][rvs][0][3].append( thr_max_log.item())
                            prev = [scale_list[i],a,round((2**7 * 2**16)/float(scale_list[i])),thr_max_log.item()]

                    self.RVSTables["data"][cwg][rvs][0][0] = torch.nn.Embedding.from_pretrained(table2.unsqueeze(1))
                    self.RVSTables["data"][cwg][rvs][0][1] = torch.nn.Embedding.from_pretrained(table3.unsqueeze(1))
                    self.RVSTables["data"][cwg][rvs][0][2] = torch.nn.Embedding.from_pretrained(table4.unsqueeze(1))
                    self.RVSTables["isBuilt"] = True
            '''print the specification table
            for i in range (4):
                for 0 in range (2):
                    print(i, 0)
                    print(self.active_tool_id)
                    print( self.CD_text[2][0][0][i], end=" ")
                    print( self.CD_text[1][0][0][i], end=" ")
                    print( self.CD_text[2][1][0][i], end=" ")
                    print( self.CD_text[1][1][0][i], end=" ")
                    print( self.CD_text[0][1][0][i], end=" ")
                    print()
            '''           


    #@timeit
    def analyzeCWG(self, decisions: Decisions):
        if not self.cwg_enabled:
            return 
        if len(self.cwgf) == 0:
            
            scale_log = decisions.get('scale_log', None)
            avg_sig = torch.mean(scale_log.float(), dim=[2, 3])
            avg_sig_sort = torch.sort(avg_sig, descending=True)
            self.cwgf = torch.zeros(scale_log.shape[1]).int()
            self.cwgf[avg_sig_sort.indices[0, :self.cnum_list[self.get_active_tool_idx()]]] = 1
            self.cwgf = self.cwgf.unsqueeze(-1).unsqueeze(-1).to(scale_log.device)
        else:
            scale_log = decisions.get('scale_log', None)
            self.cwgf = self.cwgf.to(scale_log.device)
        
    
    #@timeit
    def analyze(self, decisions: Decisions) -> Decisions:
        self.buildTables()
        self.analyzeCWG(decisions)
        scale_log = decisions.get('scale_log', None)
        h, w = scale_log.shape[-2:]
        if self.is_enabled():
            block_size = 8
            h_pad = ((h + block_size - 1) // block_size) * block_size - h
            w_pad = ((w + block_size - 1) // block_size) * block_size - w
            likely = F.pad(scale_log, (0, w_pad, 0, h_pad), value=self.pad_val)
            avgpool = torch.nn.AvgPool2d((block_size, block_size), (block_size, block_size), 0,divisor_override=1)
            
            shift = 2* log2(block_size)
            likely = avgpool(likely.float())
            likely = likely + 2**(shift - 1)
            likely = torch.bitwise_right_shift(likely.int(),int(shift))
            if block_size > 1:
                self.likely = einops.repeat(likely,
                                    'a b c d -> a b (c repeat1) (d repeat2)',
                                    repeat1=block_size,
                                    repeat2=block_size).long()
            self.likely = F.pad(self.likely, (0, -w_pad, 0, -h_pad), value=1)
            self.likely = torch.clamp (self.likely,0,len(Log2LinConvertion.table)-1)
            decisions["likely"] = deepcopy(self.likely)
            ans = Decisions()
            ans['scales'] = self.likely

            return ans

    #@timeit
    def quantize_resi(self, x: torch.Tensor, decisions: Decisions = None) -> torch.Tensor:
        self.likely = decisions.get('scales')
        if not (self.cwg_enabled or self.rvs_enabled):
            return x
        if self.cwg_enabled == False:
            t = self.RVSTables["data"][0][self.rvs_enabled][0][0]
            if self.likely.device != t.weight.device:
                t.to(device=self.likely.device)
                out = (x * t(self.likely).squeeze())/(2**self.sigma_precision)
            else:
                out = (x * t(self.likely).squeeze())>>self.sigma_precision
        else:
            t1 = self.RVSTables["data"][1][self.rvs_enabled][0][0]
            t2 = self.RVSTables["data"][2][self.rvs_enabled][0][0]
            if self.likely.device != t1.weight.device:
                t1.to(device=self.likely.device)
                t2.to(device=self.likely.device)
            t = t1(self.likely).squeeze()*self.cwgf + t2(self.likely).squeeze()*(1-self.cwgf)
            out = x * t / (2**self.sigma_precision) 
        return out

    
    #@timeit
    def quantize_scale(self, x: torch.Tensor, decisions: Decisions) -> torch.Tensor:
        self.likely = decisions.get('scales')
        if not (self.cwg_enabled or self.rvs_enabled):
            return x
        if self.cwg_enabled == False:
            t = self.RVSTables["data"][0][self.rvs_enabled][0][1]
            if self.likely.device != t.weight.device:
                t.to(device=self.likely.device)
            out = x + t(self.likely).squeeze()
        else:

            t1 = self.RVSTables["data"][1][self.rvs_enabled][0][1]
            t2 = self.RVSTables["data"][2][self.rvs_enabled][0][1]
            if self.likely.device != t1.weight.device:
                t1.to(device=self.likely.device)
                t2.to(device=self.likely.device)
            a = t1(self.likely).squeeze()
            t = t2(self.likely).squeeze()
            if str(self.likely.device) == 'cpu':
                index = self.cwgf.squeeze().bool()
                t[index,:,:] = a[index,:,:]
                out = x + t
            else:
                out = x + a*self.cwgf + t*(1-self.cwgf)
        return out
    
    #@timeit
    def dequantize_resi(self, x: torch.Tensor, decisions: Decisions = None) -> torch.Tensor:
        self.likely = decisions.get('scales')
        if not (self.cwg_enabled or self.rvs_enabled):
            return x
        if self.cwg_enabled == False:
            t = self.RVSTables["data"][0][self.rvs_enabled][0][2]
            if self.likely.device != t.weight.device:
                t.to(device=self.likely.device)
                out = (x * t(self.likely).squeeze())/(2**16)
            else:
                out = (x * t(self.likely).squeeze())>>16
        else:
            t1 = self.RVSTables["data"][1][self.rvs_enabled][0][2]
            t2 = self.RVSTables["data"][2][self.rvs_enabled][0][2]
            if self.likely.device != t1.weight.device:
                t1.to(device=self.likely.device)
                t2.to(device=self.likely.device)
            a = t1(self.likely).squeeze()
            t = t2(self.likely).squeeze()

            index = self.cwgf.squeeze().bool()
            t[index,:,:] = a[index,:,:]
            out = x*t/(2**16)
        return out


    def dequantize_scale(self, x: torch.Tensor, decisions: Decisions = None) -> torch.Tensor:
        pass