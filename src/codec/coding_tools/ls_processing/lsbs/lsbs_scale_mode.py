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

import numpy as np
import einops
import torch
import torch.nn.functional as F
from math import log2

from ..base import LSProcessingBase
from src.codec.common import Log2LinConvertion, Decisions
##
from .params import LSBSModeParams
import os
class LSBSMode(LSProcessingBase):
    """Class for handeling information about max symbol of y_hat
    """
    def __init__(self, **kwargs):
        super(LSBSMode, self).__init__(has_enabled_flag=True, **kwargs)
        self._params_LSBS_mode = LSBSModeParams()
        self.lsbs_scale_precision = 13
        self.lsbs_thr_precision = 13
        self.scaled_sigma_precision = 17
        
        self.LSBSTables = [0]*2

    def buildTables(self):
        if type(self.LSBSTables[0]) == int:

            
            self.table1 = torch.zeros_like(Log2LinConvertion.table)
            for i in range(len(Log2LinConvertion.table)):
                self.table1[i] = i
            

            for i in range(5):
                table2 = torch.zeros_like(Log2LinConvertion.table)
                table3 = torch.zeros_like(Log2LinConvertion.table)
                power = 4
                thr_min_log = -1000000
                thr = self.threshold_lsbs
                thr.append(10000000)
                scale_list0 = self.scale0_lsbs
                scale_list1 = self.scale1_lsbs
                scale_before = [0,0]
                for j in range(3):
                    
                    thr_max = thr[j]
                    thr_max = int(thr_max * (2**power))
                    thr_max_log = Log2LinConvertion.idx2log_table(torch.tensor(thr_max))
                    mask = torch.relu(self.table1-thr_min_log) - torch.relu(self.table1-thr_min_log - 1)
                    table2 += (mask * (scale_list0[j] - scale_before[0])).int()
                    table3 += (mask * (scale_list1[j] - scale_before[1])).int()
                    
                    thr_min_log = thr_max_log
                    scale_before = [scale_list0[j],scale_list1[j]]


                self.LSBSTables[0] = torch.nn.Embedding.from_pretrained(table2.unsqueeze(1))
                self.LSBSTables[1] = torch.nn.Embedding.from_pretrained(table3.unsqueeze(1))


    def export_models(self, output_dir: str, opset_version: int):
        return 
        # Almost impossible to match these tables with variables in spec.
        self.buildTables()
        data = deepcopy(self.LSBSTables)
        for i in range(len(data)):
            if isinstance(data[i], torch.nn.Embedding):
                data[i] = data[i].weight.view(-1).cpu().numpy()
        np_arr = np.array(data)
        os.makedirs(output_dir, exist_ok=True)
        for i in range(len(data)):
            assert not (np_arr[i].astype(np.int) - np_arr[i]).any()
            np.savetxt(os.path.join(output_dir, f"lsbs_table_idx{i}.csv"), np_arr[i].astype(np.int), fmt='%d', delimiter=',')
     

    def get_scale_lsbs(self, scales_hat):
        self.buildTables()
        
        _, _, h, w = scales_hat.shape
        block_size = 8
        
        h_pad = ((h + block_size - 1) // block_size) * block_size - h
        w_pad = ((w + block_size - 1) // block_size) * block_size - w
        likely = F.pad(torch.abs(scales_hat).float(), (0, w_pad, 0, h_pad), value=1411).int()
        
        avgpool = torch.nn.AvgPool2d((block_size, block_size), (block_size, block_size), 0,divisor_override=1)
        shift = log2(block_size*block_size)
        likely = avgpool(likely.float())
        likely = likely + 2**(shift - 1)
        self.likely = torch.bitwise_right_shift(likely.int(),int(shift))

        if block_size > 1:
            self.likely = einops.repeat(self.likely,
                                'a b c d -> a b (c repeat1) (d repeat2)',
                                repeat1=block_size,
                                repeat2=block_size).long()
        self.likely = F.pad(self.likely, (0, -w_pad, 0, -h_pad), value=1)
        self.likely = torch.clamp (self.likely,0,len(Log2LinConvertion.table)-1)

    def post_processing(self, x: torch.Tensor, decisions: Decisions) -> torch.Tensor:
        residual_hat = decisions.get('residual')
        denoms = self.lsbs_scale_precision
        means_full = x - residual_hat
        if decisions.has_keys_with_postfix('likely'):
            self.buildTables()
            self.likely = decisions['likely']
        else:
            self.get_scale_lsbs(torch.abs(decisions.get('scale_log')))
        if x.device != self.LSBSTables[0].weight.device:
            self.LSBSTables[0].to(device = x.device)
            self.LSBSTables[1].to(device = x.device)
        additive = (self.LSBSTables[0](self.likely).squeeze(-1) * means_full + self.LSBSTables[1](self.likely).squeeze(-1) * residual_hat)
        additive = additive/(2**denoms)
        ans = x + additive
        return ans

