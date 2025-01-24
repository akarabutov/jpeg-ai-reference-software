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

import argparse
import torch
import numpy as np
import hashlib
import os
from sklearn_extra.cluster import KMedoids
from typing import Tuple, List

class ZReduction:
    
    def __init__(self) -> None:
        self.checkpoint_field_name = 'hyper_entropy.freqs_int'
        self.checkpoint_field_quant_name = 'hyper_entropy.is_quantized'
    
    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--n_distributions", type=int, default=128,
                            help="Target number of z distributions")
        parser.add_argument("--cp_dir", type=str, default="models/VM_common",
                            help="path to a directory with checkpoints")
        return parser.parse_args()

    @staticmethod
    def get_kl_matrix(pmfs):
        probs = pmfs / pmfs.sum(axis=1, keepdims=True)
        probs_expand = np.expand_dims(probs, 1)
        return np.sum(probs_expand * np.log2(probs_expand / probs), axis=2)

    @staticmethod
    def get_md5_list(arr):
        return [hashlib.md5(x).digest() for x in arr]

    @staticmethod
    def get_unique(arr):
        return len(set(ZReduction.get_md5_list(arr)))

    def collect_freq(self, cp_dir: str):
        freqs_all = list()
        for file_name in os.listdir(cp_dir):
            if file_name.endswith(".pth"):
                original_file_name = os.path.join(cp_dir, file_name)
                assert(os.path.exists(original_file_name)) #check file exists
                loaded = torch.load(original_file_name)
                freqs_all.extend([loaded[self.checkpoint_field_name]])

        freqs_all = torch.cat(freqs_all, dim=0).cpu().numpy()        
        print(f"Checkpoints loaded. {self.get_unique(freqs_all)} unique z distributions in total.")
        
        return freqs_all
    
    
    def reduce_z_distributions(self, freqs_all: np.ndarray, n_distributions: int) -> np.ndarray:
        kl_matrix = self.get_kl_matrix(freqs_all)
        kmedoids = KMedoids(n_clusters=n_distributions, metric='precomputed', method='pam', max_iter=100, random_state=0).fit(kl_matrix.T)
        freqs_no_repeat = freqs_all[np.array([np.where(kmedoids.labels_ == i)[0][0] for i in range(n_distributions)])]
        freqs_reduced = freqs_no_repeat[kmedoids.labels_]
        print(f"Z distributions reduced to {self.get_unique(freqs_reduced)}.")
        return freqs_reduced
    
    def convert_dist_to_indexies(self, freqs_all: np.ndarray, freqs_reduced: np.ndarray) -> Tuple[List[int]]:
        fulllist_of_md5 = self.get_md5_list(freqs_all)
        redlist_of_md5 = self.get_md5_list(freqs_reduced)
        unique_MD5_list = list(set(redlist_of_md5))
        unique_list = [freqs_all[fulllist_of_md5.index(x)] for x in unique_MD5_list]
        redidx_list = [unique_MD5_list.index(x) for x in redlist_of_md5]
        return unique_list, redidx_list

    def store_reduced_tables(self, cp_dir: str, freqs_reduced: np.ndarray) -> None:
        index_start, index_end = 0, 0
        for file_name in os.listdir(cp_dir):
            if file_name.endswith(".pth"):
                full_path = os.path.join(cp_dir, file_name)
                loaded = torch.load(full_path)

                index_end += loaded[self.checkpoint_field_name].shape[0]
                loaded[self.checkpoint_field_name][:] = torch.from_numpy(freqs_reduced[index_start : index_end])
                loaded[self.checkpoint_field_quant_name][0] = True
                index_start = index_end

                torch.save(loaded, full_path)
                print(f"Stored the new distribution to a file {full_path}")
                
    def store_reduced_indexies(self, cp_dir: str, unique_list: List[int], redidx_list: List[int]) -> None:
        np.savetxt(os.path.join(cp_dir, "unique_z_distributions.csv"), np.array(unique_list), fmt='%d', delimiter=',')

        index_start, index_end = 0, 0
        for file_name in os.listdir(cp_dir):
            if file_name.endswith(".pth"):
                full_path = os.path.join(cp_dir, file_name)
                loaded = torch.load(full_path)
                index_end += loaded[self.checkpoint_field_name].shape[0]
                full_path = os.path.join(cp_dir, file_name.replace('pth', 'csv'))
                np.savetxt(full_path, np.array(redidx_list[index_start : index_end]), fmt='%d', delimiter=',')
                index_start = index_end

                print(f"Stored indexies of the new distributions to a file {full_path}")
                
    def process_checkpoints(self, n_distributions: int, cp_dir: str):
        freqs_all = self.collect_freq(cp_dir)
        print("Reducing z distributions..")
        freqs_reduced = self.reduce_z_distributions(freqs_all, n_distributions)
        self.store_reduced_tables(cp_dir, freqs_reduced)
    

if __name__ == "__main__":
    zr = ZReduction()
    args = vars(zr.get_args())
    zr.process_checkpoints(**args)
