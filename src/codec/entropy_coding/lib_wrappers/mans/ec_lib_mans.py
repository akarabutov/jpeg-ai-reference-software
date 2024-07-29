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

import os
import torch
import numpy as np
from typing import List
from . import ans
from .custom_prob_wrapper import CustomProbWrapper
from .sgt_prob_wrapper import SgtProbWrapper
from .bypass_prob_wrapper import BypassProbWrapper
from ..ec_lib_base import ECLibBaseWithThread
import numpy as np

from .utils import (get_probs_gaussian, 
                    get_pmf_matrix, 
                    get_outbound_values, 
                    get_cdf_matrix, 
                    get_encode_transitions, 
                    get_state_maps, 
                    get_decode_transitions)

class ECLibMans(ECLibBaseWithThread):
    def __init__(
            self,
            *args,
            **kwargs):
        """
 
        Args:
            collectors:
            io_stream: bit_stream for input/output
            coder_type:
            lib_name:
            debug:
            debug_start:
        """
        super(ECLibMans, self).__init__('ECLibMans', *args, **kwargs)
        self.ae_args = args
        self.ae_kwargs = kwargs
        self.backend = None
        self.compressed_size = 0
        pdf_r=[[255],
               [255],
               [255],
               [251,2,2],
               [245,5,5],
               [231,12,12],
               [211,22,22],
               [188,33,34],
               [157,48,48,1,1],
               [134,57,56,4,4],
               [111,61,61,10,10,1,1],
               [94,62,61,17,17,2,2],
               [77,58,58,24,24,6,6,1,1],
               [63,52,52,29,29,11,11,3,3,1,1],
               [52,46,46,31,30,16,16,6,6,2,2,1,1],
               [43,39,39,30,30,19,19,10,10,5,5,2,2,1,1],
               [35,33,33,28,28,21,21,13,13,8,8,4,4,2,2,1,1],
               [29,28,28,25,25,20,20,15,15,10,10,7,7,4,4,2,2,1,1,1,1],
               [24,23,23,21,21,19,18,15,15,12,12,9,9,6,6,4,4,3,3,2,2,1,1,1,1],
               [19,19,19,18,18,17,17,15,15,12,12,10,10,8,8,6,6,4,4,3,3,2,2,1,1,1,1,1,1,1,1],
               [17,16,16,16,16,15,15,13,13,12,12,10,10,9,9,7,7,6,6,4,4,3,3,2,2,2,2,1,1,1,1,1,1,1,1],
               [14,13,13,13,13,12,12,12,12,11,11,10,10,9,9,8,8,6,6,5,5,4,5,4,4,3,3,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1],
               [11,11,11,11,11,10,10,10,10,10,10,9,9,8,8,7,7,7,7,6,6,5,5,5,5,4,4,3,3,3,3,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
               [9,9,9,9,9,9,9,8,8,8,8,8,8,7,7,7,7,6,6,6,6,5,5,5,5,4,4,4,4,4,4,3,3,3,3,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
               [7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,5,5,5,5,5,5,5,5,4,4,4,4,4,4,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
               [6,6,6,6,6,6,6,6,6,5,6,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
               [5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
               [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
               [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
               [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
               [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
              ]
        self.pdf_r = pdf_r        
        self.quant_params = [kwargs.get('quant_start', 0.11),
                             kwargs.get('quant_end', 54.82),
                             kwargs.get('quant_count', 32)]
        self.quant_values = [kwargs.get('quant_min_val', -128),
                             kwargs.get('quant_max_val', 128)]
        self.cache_file = os.path.join(os.path.dirname(__file__), 'cache.pt')
        if kwargs.get('rebuild_ae_cache', False) and os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        self.load_cache()
        
    def init_quant_params(self):     
        """Initialization of quantized parameters
        """
        bound_table_r=[1,1,1,2,2,2,2,2,3,3,4,4,5,6,7,8,9,11,13,16,18,22,26,32,38,46,55,65,77,92,106,128]

        
        bounds = np.array(bound_table_r, dtype=np.int64)
        pmf_matrix = np.zeros([len(self.pdf_r), self.quant_values[1] - self.quant_values[0]], dtype=np.int64)
        for i in range(len(self.pdf_r)):
            pmf_matrix[i, :len(self.pdf_r[i])] = self.pdf_r[i]
            pmf_matrix[i, bound_table_r[i] * 2 - 1] = 1

        cdf_matrix = get_cdf_matrix(pmf_matrix)
        self.encode_transitions = get_encode_transitions(pmf_matrix, cdf_matrix)
        self.state_maps = get_state_maps(cdf_matrix)
        self.decode_transitions = get_decode_transitions(pmf_matrix, cdf_matrix)
        self.bounds = bounds.astype(np.uint8)

    def store_cache(self):
        """Store quantized data to a cache file
        """
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        cache_data = dict()
        cache_data['encode_transition_file'] = self.encode_transitions
        cache_data['decode_transition_file'] = self.decode_transitions
        cache_data['bound_file'] = self.bounds
        cache_data['state_map_file'] = self.state_maps
        cache_data['quant_params'] = self.quant_params
        cache_data['quant_values'] = self.quant_values
        torch.save(cache_data, self.cache_file)
            
    def load_cache(self):
        """Load quantized parameters from a file 'cache.pt'
        """
        init_quant_params_data = False
        if os.path.exists(self.cache_file):
            cache_data = torch.load(self.cache_file)
            if cache_data['quant_params'] == self.quant_params and cache_data['quant_values'] == self.quant_values:
                self.encode_transitions = cache_data['encode_transition_file']
                self.decode_transitions = cache_data['decode_transition_file']
                self.bounds = cache_data['bound_file']
                self.state_maps = cache_data['state_map_file']
            else:
                init_quant_params_data = True
        else:
            init_quant_params_data = True
        
        if init_quant_params_data:
            self.init_quant_params()
            self.store_cache()

 
    # ##################################################################################################################
    #  __init__ methods
    # ##################################################################################################################
    def _init_prob_wrappers(self):
        self.prob_wrappers = {
            'Custom': CustomProbWrapper(self.backend),
            'Bypass': BypassProbWrapper(self.backend),
            'Sgt': SgtProbWrapper(self.backend),
        }

    # ##################################################################################################################
    #  encode/decode methods    

    def decode_init(self, input_mem: np.ndarray):
        super().decode_init(input_mem)
        self.load_cache()
        threads_sizes = np.array(self.get_threads_sizes(), dtype=np.uint32)
        threads_end_offsets = np.cumsum(threads_sizes).astype(np.uint32)
        self.backend = ans.ANSDecoder(input_mem, threads_end_offsets)
        self.backend.set_sgm_transitions(self.decode_transitions.data, self.bounds.data) #, self.state_maps.data)
        self._init_prob_wrappers()

    def decode_term(self):
        self.backend = None


    def encode_init(self, mem: np.ndarray):
        super().encode_init(mem)
        self.load_cache()
        self.mem = mem
        self.threads_sizes = np.zeros([self.num_threads], dtype=np.uint32)
        self.backend = ans.ANSEncoder(len(mem), self.num_threads)
        self.backend.set_sgm_transitions(self.encode_transitions.data, self.bounds.data, self.state_maps.data)
        self._init_prob_wrappers()
        
        
    def encode_term(self) -> int:
        tot_size = self.backend.close()
        self.total_bits[0] = torch.tensor(tot_size) * 8
        self.mem = self.mem[:tot_size]
        self.backend.get_memory(self.mem)
        self.backend.get_thread_sizes(self.threads_sizes)
        self.set_threads_sizes(self.threads_sizes.tolist())
        return self.total_bits[0]

    # ##################################################################################################################
    #  service methods
    # ##################################################################################################################

    def store_parameters(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        #np.savetxt('transaction_table.csv', self.decode_transitions, fmt='%d', delimiter=',')
        transition_table_nBits = np.zeros_like(self.decode_transitions)
        transition_table_stateNext  = np.zeros_like(self.decode_transitions)
        transition_table_symbol  = np.zeros_like(self.decode_transitions)
        for c in range(len(self.decode_transitions)):
            for i in range(len(self.decode_transitions[c])):
                transition_table_nBits[c,i] = (self.decode_transitions[c,i] >> 24) & 255
                transition_table_stateNext[c,i] = (self.decode_transitions[c,i] >> 16) & 255
                transition_table_symbol[c,i] = (self.decode_transitions[c,i] >> 16) & 65535
        np.savetxt(os.path.join(output_dir, 'transition_table_nBits.csv'), transition_table_nBits, fmt='%d', delimiter=',')
        np.savetxt(os.path.join(output_dir, 'transition_table_symbol.csv'), transition_table_symbol, fmt='%d', delimiter=',')
        np.savetxt(os.path.join(output_dir, 'transition_table_stateNext.csv'), transition_table_stateNext, fmt='%d', delimiter=',')        
        np.savetxt(os.path.join(output_dir, 'bounds.csv'), self.bounds, fmt='%d', delimiter=',')
        with open(os.path.join(output_dir, 'pdf.csv'), "w") as f:
            for l in self.pdf_r:
                f.write(",".join([str(x) for x in l]) + '\n')
        #np.savetxt(os.path.join(output_dir, 'pdf.csv'), self.pdf_r, fmt='%d', delimiter=',')
        
    def get_z_transactions(self, cdfs: np.ndarray):
        ans = np.zeros([256], dtype=np.int)
        self.backend.get_z_transactions(cdfs, ans)
        return ans