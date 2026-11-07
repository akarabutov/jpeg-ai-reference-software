# The copyright in this software is being made available under the BSD
# License, included below. This software may be subject to other third party
# and contributor rights, including patent rights, and no such rights are
# granted under this license.
#
# Copyright (c) 2010-2026, ITU/ISO/IEC
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
# DISCLAIMER
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

"""Helpers shared by the Lightning training of CCS: meters, the result table and the
statistics of the data collection stage."""

import json

import numpy as np
import torch
from prettytable import PrettyTable

from src.train.CCS.utils import AverageMeter

__all__ = [
    'ResultPrinter', 'create_meters', 'update_meters', 'print_meters', 'has_nan', 'get_thres',
    'set_clip_thres', 'create_collection_stat', 'COLLECTION_KEYS'
]

COLLECTION_KEYS = ('E1B', 'E2B', 'E3B', 'E4B', 'E5B',
                   'E1H', 'E2H', 'E3H', 'E4H', 'E5H',
                   'HE1', 'HE2', 'HE3', 'HE4', 'HE5')


def has_nan(tensor_list):
    ret = False
    for tensor in tensor_list:
        if torch.isnan(tensor).any().item():
            ret = True
    return ret


class ResultPrinter:
    default_headers = (
        'Time',
        'Data',
        'Rate',
        'Hyper_rate',
        'Y_PSNR',
        'U_PSNR',
        'V_PSNR',
        'Loss',
    )

    def __init__(self, epoch_str='', headers=None):
        if headers is None:
            headers = list(self.default_headers)  # copy
        self._headers = headers
        self._table = PrettyTable()
        self._table.field_names = [epoch_str] + self._headers

    def insert_row(self, meters, call_str='', meter_attribute='val'):
        if meter_attribute == 'val':
            results = {meter.name: meter.val for meter in meters.values()}
        elif meter_attribute == 'avg':
            results = {meter.name: meter.avg for meter in meters.values()}
        else:
            assert False, 'Not Implemented'

        self.update(results, call_str)

    def update(self, results, call_str=''):
        row = [call_str] + [f'{results[header]:.4f}' for header in self._headers]
        self._table.add_row(row)

    def print(self):
        print(self._table)
        
    @staticmethod
    def get_dict_from_row(field_names, row):
        ans = dict()
        for k,v in zip(field_names, row):
            ans[k] = v
        return ans
        
    def get_dict(self):
        """Generate dictionary based on a table
        """
        ans = list()
        f = self._table.field_names
        for row in self._table._rows:
            ans.append(self.get_dict_from_row(f, row))
        return ans
        
    def store(self, json_path: str):
        """Store the table to a json file

        Args:
            json_path (str): path to an output json file
        """
        output_arr = self.get_dict()
        with open(json_path, 'w') as f:
            json.dump(output_arr, f)
        
        


def update_meters(batch_size, meters, kwargs):
    for key, val in kwargs.items():
        if key != 'rec_data':
            if key != 'Time':
                meters[key].update(val, batch_size)
            else:
                meters[key].update(val)


def create_meters(
        keys=['Time', 'Data', 'Rate', 'Hyper_rate', 'Y_PSNR', 'U_PSNR', 'V_PSNR', 'Loss']):
    meters = [AverageMeter(key) for key in keys]
    meters_dict = {meter.name: meter for meter in meters}
    return meters_dict


def print_meters(meters_dict, keys=['Time'], prefix=''):
    row = prefix
    for key in keys:
        if key != 'Time':
            row += f'{key} {meters_dict[key].val:.4e} '
        else:
            row += f'{key} {meters_dict[key].val:.4f}({meters_dict[key].avg:.4f}) '
    print(row)


def idx_to_num(idx, min, max, bin_num):
    intv = (max - min) / bin_num
    num = min + idx * intv
    return num


def get_thres(stat, remv_num):
    stat_thres = dict.fromkeys(stat.keys(), {})
    for layer in stat:
        hist = np.array(stat[layer]['StaList'])
        if hist.shape[0] == 0:
            continue
        c, bins = hist.shape

        stat_thres[layer] = dict.fromkeys(('MaxList', 'MinList'), [])
        stat_thres[layer]['MaxList'] = np.zeros(c)                
        stat_thres[layer]['MinList'] = np.zeros(c)                

        for ch in range(c):
            total_num = np.sum(hist[ch])
            hist_cum = np.cumsum(hist[ch])
            num_upper = total_num - remv_num
            num_lower = remv_num

            upper_idx = np.searchsorted(hist_cum, num_upper, side='left')
            lower_idx = np.searchsorted(hist_cum, num_lower, side='right')
            stat_thres[layer]['MaxList'][ch] = min(idx_to_num(upper_idx + 1, stat[layer]['MinList'][ch], stat[layer]['MaxList'][ch], 1000), stat[layer]['MaxList'][ch])
            stat_thres[layer]['MinList'][ch] = max(idx_to_num(lower_idx, stat[layer]['MinList'][ch], stat[layer]['MaxList'][ch], 1000), stat[layer]['MinList'][ch])
        
        stat_thres[layer]['MaxList'] = stat_thres[layer]['MaxList'].tolist()
        stat_thres[layer]['MinList'] = stat_thres[layer]['MinList'].tolist()

    return stat_thres


def set_clip_thres(m, stat, key_list):
    m.clip_thres = dict()
    for k, v in stat.items():
        if k in key_list:
            m.clip_thres[k] = v 


def create_collection_stat():
    """Empty statistics of the data collection stage (one per component: Y or UV)."""
    return {key: {'MaxList': [], 'MinList': [], 'StaList': []} for key in COLLECTION_KEYS}
