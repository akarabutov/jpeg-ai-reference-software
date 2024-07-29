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

import numpy as np
from scipy.special import erf

def get_cdf_gaussian(val, scale):
    return (1 + erf(val / scale / np.sqrt(2))) / 2


def get_probs_gaussian(quantized_stds, quantized_values):
    quantized_values = quantized_values.reshape(1, -1)
    quantized_stds = quantized_stds.reshape(-1, 1)
    q_lowers, q_uppers = quantized_values - 0.5, quantized_values + 0.5
    return get_cdf_gaussian(q_uppers, quantized_stds) - get_cdf_gaussian(q_lowers, quantized_stds)


def get_outbound_values(probs, threshold=1/2**11):
    return probs.shape[1] // 2 - np.clip(np.argmax(probs > threshold, axis=1) - 1, 0, None)


def get_sequence(length=256):
    mid = length // 2
    result = np.zeros(length, dtype=np.int64)
    result[::2] = np.arange(mid, length)
    result[1::2] = np.arange(mid)[::-1]
    return result


def get_inverse_sequence(length=256):
    mid = length // 2
    result = np.zeros(length, dtype=np.int64)
    result[:mid] = np.arange(mid, 0, -1) * 2 - 1
    result[mid:] = np.arange(mid) * 2
    return result


def get_pmf_matrix(probs, bounds, mass_bits=8):
    pmfs_matrix = np.zeros(probs.shape, dtype=np.int64)
    probs = probs[:, get_sequence(pmfs_matrix.shape[1])]
    probs = probs / np.cumsum(probs, axis=1)
    for row, bound in enumerate(bounds):
        total_pmf_left = 1 << mass_bits
        for col in reversed(range(bound + bound)):
            pmfs_matrix[row, col] = np.maximum(np.round(probs[row, col] * total_pmf_left), 1)
            total_pmf_left -= pmfs_matrix[row, col]
    return pmfs_matrix


def get_cdf_matrix(pmf_matrix):
    cdf_matrix = np.zeros([pmf_matrix.shape[0], pmf_matrix.shape[1] + 1], dtype=np.int64)
    cdf_matrix[:, 1:] = np.cumsum(pmf_matrix, axis=1)
    return cdf_matrix


def get_clz(length, mass_bits=8):
    return (mass_bits + 1 - np.log2(np.arange(length) + 0.5)).astype(np.int64)


def get_delta_bits(mass_bits=8):
    num_bits = get_clz(1 << mass_bits, mass_bits=mass_bits)
    return (num_bits + 1 << mass_bits) - (np.arange(1 << mass_bits) << num_bits)


def get_encode_transitions(pmfs, cdfs, mass_bits=8):
    adders = (cdfs[:, :-1] - pmfs).astype(np.uint8)
    delta_bits = get_delta_bits(mass_bits)[pmfs]
    transitions = ((delta_bits << 16) | adders).astype(np.uint32)
    return transitions[:, get_inverse_sequence(pmfs.shape[1])].copy()


def get_state_maps(cdfs, mass_bits=8):
    state_maps = np.zeros([cdfs.shape[0], 1 << mass_bits], dtype=np.uint8)
    cdf_first, cdf_second = cdfs - (cdfs >> 1), (cdfs >> 1) + 128
    cdf_diff_first = cdf_first[:, 1:] - cdf_first[:, :-1]
    for i in range(cdfs.shape[0]):
        for j in range(cdfs.shape[1] - 1):
            state_maps[i, cdfs[i, j]: cdfs[i, j] + cdf_diff_first[i, j]] = np.arange(cdf_first[i, j], cdf_first[i, j + 1])
            state_maps[i, cdfs[i, j] + cdf_diff_first[i, j]: cdfs[i, j + 1]] = np.arange(cdf_second[i, j], cdf_second[i, j + 1])
    return state_maps


def get_decode_transitions(pmfs, cdfs, mass_bits=8):
    num_distributions = pmfs.shape[0]
    sequences = get_sequence(pmfs.shape[1]) - pmfs.shape[1] // 2
    symbols = np.zeros([num_distributions, 1 << mass_bits], dtype=np.int64)
    state_next = np.zeros([num_distributions, 1 << mass_bits], dtype=np.int64)
    cdf_first, cdf_second = cdfs - (cdfs >> 1), (cdfs >> 1) + 128
    cdf_diff_first = cdf_first[:, 1:] - cdf_first[:, :-1]

    for i in range(num_distributions):
        col, cdf_curr = 0, 0
        while cdf_curr < (1 << mass_bits):
            pmf, cdf_next = pmfs[i, col], cdfs[i, col + 1]
            symbols[i, cdf_first[i, col]: cdf_first[i, col + 1]] = sequences[col]
            symbols[i, cdf_second[i, col]: cdf_second[i, col + 1]] = sequences[col]
            state_next[i, cdf_first[i, col]: cdf_first[i, col + 1]] = np.arange(pmf, pmf + cdf_diff_first[i, col])
            state_next[i, cdf_second[i, col]: cdf_second[i, col + 1]] = np.arange(pmf + cdf_diff_first[i, col], pmf + pmf)
            col, cdf_curr = col + 1, cdf_next

    num_bits = get_clz(1 << mass_bits + 1, mass_bits)[state_next]
    state_next = (state_next << num_bits) ^ (1 << mass_bits)
    return ((num_bits << 24) | (state_next << 16) | (symbols & 65535)).astype(np.uint32)


def normalize_z(pmfs_z):
    pmfs_z_cumsum = np.cumsum(pmfs_z, axis=1)
    pmfs_z_norm = ((pmfs_z_cumsum * 255 + (pmfs_z_cumsum[:, -1:] >> 1)) // pmfs_z_cumsum[:, -1:]).astype(np.uint8)
    return pmfs_z_norm
