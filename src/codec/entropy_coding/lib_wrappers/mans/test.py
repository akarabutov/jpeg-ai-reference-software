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

import os
import torch
import numpy as np
import json
from json import JSONEncoder
from . import ans
import hashlib

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
class TestANS(unittest.TestCase):
    
    
    def get_hash(tensor: np.ndarray) -> str:
        return hashlib.md5(tensor).hexdigest()
        
    def load_cache(self, cache_file=None):
        """Load quantized parameters from a file 'cache.pt'
        """
        if cache_file is None:
            cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache.pt')
        if os.path.exists(cache_file):
            return torch.load(cache_file)
        else:
            assert 0
    
    def init_encoder(self, memory_capacity, num_threads):
        encoder = ans.ANSEncoder(memory_capacity, num_threads)
        cache = self.load_cache()
        encoder.set_sgm_transitions(cache['encode_transition_file'].data, cache['bound_file'].data, cache['state_map_file'].data)
        return encoder

    def init_decoder(self, memory, thread_sizes):
        decoder = ans.ANSDecoder(memory, thread_sizes)
        cache = self.load_cache()
        decoder.set_sgm_transitions(cache['decode_transition_file'].data, cache['bound_file'].data)
        return decoder
    
    @staticmethod
    def normalize_z(pmfs_z_tensor):
        pmfs_z_cumsum = np.cumsum(pmfs_z_tensor, axis=0)
        pmfs_z_norm = ((pmfs_z_cumsum * 255 + (pmfs_z_cumsum[-1:] >> 1)) // pmfs_z_cumsum[-1:]).astype(np.uint8)
        return pmfs_z_norm
    
    
    def gen_factorized_pmf(self, values_count:int=63):
        cdf = np.random.randint(4, 10, [values_count], dtype=np.int)
        pmf = self.normalize_z(cdf)
        pmf = np.expand_dims(pmf, 0)
        return pmf
        
    
    def test_factorize(self):
        MAX_SEQ_LEN = 100000
        MEMORY_SIZE = MAX_SEQ_LEN * 16
        MAX_VALUE=62
        for num_thread in [1,2,4,8,16]:
            for _ in range(1000):
                SEQ_LEN = np.random.randint(num_thread, MAX_SEQ_LEN)
                pmf = self.gen_factorized_pmf(MAX_VALUE)
                values = np.random.randint(0, MAX_VALUE, [SEQ_LEN]).astype(np.int8)
                values = np.expand_dims(values, 0)
                values_encode, values_decode = values.copy(), np.empty_like(values)

                encoder = self.init_encoder(MEMORY_SIZE, num_thread)
                encoder.encode_factorize(pmf, values_encode)
                bs_size = encoder.close()
                
                self.memory = np.empty([bs_size], dtype=np.uint8)
                self.thread_sizes = np.empty([num_thread], dtype=np.uint32)
                encoder.get_memory(self.memory)
                encoder.get_thread_sizes(self.thread_sizes)
                
                offsets = np.cumsum(self.thread_sizes).astype(np.uint32)
                decoder = self.init_decoder(self.memory, offsets)
                decoder.decode_factorize(pmf, values_decode)
                is_ok = (values == values_decode).all()
                if not is_ok:
                    arr = {
                        'values': values,
                        'thread_sizes': self.thread_sizes
                    }
                    with open('factorized_err_arr.json', 'w') as f:
                        json.dump(arr, f, cls=NumpyArrayEncoder)
                self.assertTrue(is_ok, msg=f"Test failed for threads {num_thread}")               
    
    def test_sgm(self):
        MAX_SEQ_LEN = 100000
        MEMORY_SIZE = MAX_SEQ_LEN * 10
        SIGMA_LEN=32
        power = 15
        for num_thread in [1,2,4,8,16]:
            for num_test in range(1000):
                SEQ_LEN = np.random.randint(num_thread, MAX_SEQ_LEN)
                np.random.seed(num_test)
                sigma_index = np.random.randint(0, SIGMA_LEN-1, [SEQ_LEN]).astype(np.uint8)
                values = np.random.randint(1-2**power, -1 + 2**power, [SEQ_LEN]).astype(np.int16)
                values_encode, values_decode = values.copy(), np.empty_like(values)

                mask = np.ones(list(sigma_index.shape), dtype=np.bool)
                encoder = self.init_encoder(MEMORY_SIZE, num_thread)
                encoder.encode_sgm(sigma_index, values_encode, mask)
                bs_size = encoder.close()

                self.memory = np.empty([bs_size], dtype=np.uint8)
                self.thread_sizes = np.empty([num_thread], dtype=np.uint32)
                encoder.get_memory(self.memory)
                encoder.get_thread_sizes(self.thread_sizes)

                self.offsets = np.cumsum(self.thread_sizes).astype(np.uint32)
                decoder = self.init_decoder(self.memory, self.offsets)
                decoder.decode_sgm(sigma_index, values_decode, mask)
                is_ok = (values == values_decode).all()
                if not is_ok:
                    arr = {
                        'sigma_index': sigma_index,
                        'values_encode': values_encode,
                        'mask': mask,
                        'thread_sizes': self.thread_sizes
                    }
                    with open('sgm_err_arr.json', 'w') as f:
                        json.dump(arr, f, cls=NumpyArrayEncoder)
                    
                self.assertTrue(is_ok, msg=f"Test failed for threads {num_thread} and range [{1-(2**power)}, {-1 + 2**power}] {(values == values_decode).all()}")
  
    def _test_selected_case(self):
        import json
        import numpy as np
        MEMORY_SIZE = 1908160 * 10
        num_thread = 8

        if os.path.exists('sgm_err_arr.json'):
            with open("sgm_err_arr.json", "r") as file:
                content = json.load(file)
            
            sigma_index = np.array(content['sigma_index']).astype(np.uint8)
            values = np.array(content['values_encode']).astype(np.int16)
            mask = np.array(content['mask']).astype(np.bool)
            values_encode, values_decode = values.copy(), np.empty_like(values)

            encoder = self.init_encoder(MEMORY_SIZE, num_thread)
            encoder.encode_sgm(sigma_index, values_encode, mask)
            bs_size = encoder.close()

            self.memory = np.empty([bs_size], dtype=np.uint8)
            self.thread_sizes = np.empty([num_thread], dtype=np.uint32)
            encoder.get_memory(self.memory)
            encoder.get_thread_sizes(self.thread_sizes)
            
            self.offsets = np.cumsum(self.thread_sizes).astype(np.uint32)
            decoder = self.init_decoder(self.memory, self.offsets)
            decoder.decode_sgm(sigma_index, values_decode, mask)
            is_ok = (values == values_decode).all()
            
            self.assertTrue(is_ok, msg=f"Selected case test failed")  
                            
if __name__ == "__main__":
    unittest.main()
