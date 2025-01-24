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

import unittest
import torch
from .utils import ContextUtils


class SepContextTest(unittest.TestCase):
    def test_down_n_up_shuffle_4stages(self):
        init_data = torch.tensor([[[[0,2,0,2], [3,1,3,1], [0,2,0,2], [3,1,3,1]],
              [[4,6,4,6], [7,5,7,5], [4,6,4,6], [7,5,7,5]]]])
        output_data = torch.tensor(
            [[[[0, 0], [0, 0]],
              [[1, 1], [1, 1]],
              [[2, 2], [2, 2]],
              [[3, 3], [3, 3]],
              [[4, 4], [4, 4]],
              [[5, 5], [5, 5]],
              [[6, 6], [6, 6]],
              [[7, 7], [7, 7]]]]
        )

        # test down shuffle proc
        # shuld be split in channel dim first
        initdata_slice1, init_data_slice2 = torch.chunk(init_data, dim=1,chunks=2)
        slice1_ctx = ContextUtils.down_shuffle(initdata_slice1)
        slice2_ctx = ContextUtils.down_shuffle(init_data_slice2)
        slice1_ctx_cat = torch.cat(slice1_ctx,dim=1)
        slice2_ctx_cat = torch.cat(slice2_ctx,dim=1)
        a = torch.cat((slice1_ctx_cat,slice2_ctx_cat), dim=1 )

        #test up_shuffle proc
        up_shuffle_slice1 = ContextUtils.up_shuffle(slice1_ctx)
        up_shuffle_slice2 = ContextUtils.up_shuffle(slice2_ctx)
        cat_arr = torch.cat((up_shuffle_slice1,up_shuffle_slice2),dim=1)

        self.assertEqual(a.numpy().tolist(), output_data.numpy().tolist(), "Tensor disassembled incorrectly")

        self.assertEqual(init_data.numpy().tolist(), cat_arr.numpy().tolist(), "Tensor assembled incorrectly")
        
if __name__ == '__main__':
    unittest.main()