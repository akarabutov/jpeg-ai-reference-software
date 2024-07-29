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

import numpy as np
from typing import List
from .ec_direct import EcLibDirect

class TestDirectTask(unittest.TestCase):
    
    def test_direct(self):
        ARR_SIZE = 10000
        for b in range(1,31):
            arr = np.zeros([ARR_SIZE], dtype=np.int8)
            data_size = 8 * ARR_SIZE // b 
            enc_data = np.random.randint(0, 2**b, data_size, dtype=np.uint)
            dec_data = np.zeros_like(enc_data)
            
            enc = EcLibDirect(arr)
            for i in range(data_size):
                enc.write_bits(enc_data[i], b)
            dec = EcLibDirect(arr)
            for i in range(data_size):
                output_len, dec_data[i] = dec.read_bits(b)
                self.assertEqual(output_len, b, "Incorrect number of read symbols")
            #self.assertEqual(enc_data, dec_data)
            self.assertTrue((enc_data == dec_data).all(), msg=f"Check for b = {b}")
            self.assertEqual(enc.get_pointer_pos(), dec.get_pointer_pos(), msg = "Position on enc/dec")

    
if __name__ == "__main__":
    unittest.main()