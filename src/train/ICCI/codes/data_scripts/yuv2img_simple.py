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

import math
import os
import re
from functools import partial

import numpy as np
import torch
from torch.nn import functional as F

def convert_chroma_420_to_444(data: torch.Tensor,
                              mode='bicubic',
                              new_size=None) -> torch.Tensor:
    assert new_size
    data = data.unsqueeze(0).unsqueeze(0)
    upsample_data = F.interpolate(data, size=new_size, mode=mode, align_corners=True)
    return upsample_data.squeeze(0).squeeze(0).numpy()

def yuv_420_to_444(data, new_size):

    data = torch.from_numpy(np.ascontiguousarray(data)).float()
    data_up =  convert_chroma_420_to_444(data, mode='bicubic', new_size=new_size)
    data_up = (data_up + 0.5).clip(0., 255.)
    return np.floor(data_up).astype(np.uint8)

def rgb2yuv(rgb):
    """rgb2yuv

    Args:
        rgb: shape=[..., C, *, *]
        yuv_type:

    Returns:
        yuv_list: a list of channel instances.
    """
    params = {
        'Kr': 0.2126,
        'Kg': 0.7152,
        'Kb': 0.0722,
        'Kby': 1.8556,
        'Kry': 1.5748}

    r = rgb[..., 0, :, :]
    g = rgb[..., 1, :, :]
    b = rgb[..., 2, :, :]

    y = params['Kr'] * r + params['Kg'] * g + params['Kb'] * b
    u = (b - y) / params['Kby'] + 0.5
    v = (r - y) / params['Kry'] + 0.5

    yuv = np.stack((y,u,v), axis=0)
    # yuv = np.floor(yuv)
    # yuv = yuv.astype(np.uint8)
    return yuv

def readyuv420(filename, startframe=0, totalframe=1):

    #BasketballDrill_832x480_50fps_8bit_420.yuv
    #0.002_BasketballDrill_832x480_50fps_8bit_420.yuv
    file_basename = os.path.basename(filename)
    Frame_size = re.findall('_(\d+?)x(\d+?)_', file_basename)
    W = int(Frame_size[0][0])
    H = int(Frame_size[0][1])

    uv_H = H // 2
    uv_W = W // 2

    if '8bit' in file_basename:
        bitdepth = 8
        Y = np.zeros((totalframe, H, W), np.uint8)
        U = np.zeros((totalframe, uv_H, uv_W), np.uint8)
        V = np.zeros((totalframe, uv_H, uv_W), np.uint8)
    elif '10bit' in file_basename:
        bitdepth = 10
        Y = np.zeros((totalframe, H, W), np.uint16)
        U = np.zeros((totalframe, uv_H, uv_W), np.uint16)
        V = np.zeros((totalframe, uv_H, uv_W), np.uint16)

    bytes2num = partial(int.from_bytes, byteorder='little', signed=False)

    bytesPerPixel = math.ceil(bitdepth / 8)
    seekPixels = startframe * H * W * 3 // 2
    fp = open(filename, 'rb')
    fp.seek(bytesPerPixel * seekPixels)

    for i in range(totalframe):
        for m in range(H):
            for n in range(W):
                if bitdepth == 8:
                    pel = bytes2num(fp.read(1))
                    Y[i, m, n] = np.uint8(pel)
                elif bitdepth == 10:
                    pel = bytes2num(fp.read(2))
                    Y[i, m, n] = np.uint16(pel)

        for m in range(uv_H):
            for n in range(uv_W):
                if bitdepth == 8:
                    pel = bytes2num(fp.read(1))
                    U[i, m, n] = np.uint8(pel)
                elif bitdepth == 10:
                    pel = bytes2num(fp.read(2))
                    U[i, m, n] = np.uint16(pel)

        for m in range(uv_H):
            for n in range(uv_W):
                if bitdepth == 8:
                    pel = bytes2num(fp.read(1))
                    V[i, m, n] = np.uint8(pel)
                elif bitdepth == 10:
                    pel = bytes2num(fp.read(2))
                    V[i, m, n] = np.uint16(pel)
    return Y, U, V


def writeyuv420(Y, U, V, filename):
    file_basename = os.path.basename(filename)
    if '8bit' in file_basename:
        bitdepth = 8
    elif '10bit' in file_basename:
        bitdepth = 10
    totalframe, H, W = np.shape(Y)
    uv_H = np.shape(U)[1]
    uv_W = np.shape(U)[2]
    fp = open(filename, 'wb')

    for i in range(totalframe):
        for m in range(H):
            for n in range(W):
                if bitdepth == 8:
                    pel = bytes(np.array([Y[i, m, n]], dtype=np.uint8))
                    fp.write(pel)
                elif bitdepth == 10:
                    pel = bytes(np.array([Y[i, m, n]], dtype=np.uint16))
                    fp.write(pel)

        for m in range(uv_H):
            for n in range(uv_W):
                if bitdepth == 8:
                    pel = bytes(np.array([U[i, m, n]], dtype=np.uint8))
                    fp.write(pel)
                elif bitdepth == 10:
                    pel = bytes(np.array([U[i, m, n]], dtype=np.uint16))
                    fp.write(pel)

        for m in range(uv_H):
            for n in range(uv_W):
                if bitdepth == 8:
                    pel = bytes(np.array([V[i, m, n]], dtype=np.uint8))
                    fp.write(pel)
                elif bitdepth == 10:
                    pel = bytes(np.array([V[i, m, n]], dtype=np.uint16))
                    fp.write(pel)


def readyuv420_single(filename, startframe=0, totalframe=1):
    #BasketballDrill_832x480_50fps_8bit_420_00.yuv
    #0.002_BasketballDrill_832x480_50fps_8bit_420_00.yuv
    file_basename = os.path.basename(filename)
    Frame_size = re.findall('_(\d+?)x(\d+?)_', file_basename)
    W = int(Frame_size[0][0])
    H = int(Frame_size[0][1])

    uv_H = H // 2
    uv_W = W // 2

    if '8bit' in file_basename:
        bitdepth = 8
        Y = np.zeros((totalframe, H, W), np.uint8)
        U = np.zeros((totalframe, uv_H, uv_W), np.uint8)
        V = np.zeros((totalframe, uv_H, uv_W), np.uint8)
    elif '10bit' in file_basename:
        bitdepth = 10
        Y = np.zeros((totalframe, H, W), np.uint16)
        U = np.zeros((totalframe, uv_H, uv_W), np.uint16)
        V = np.zeros((totalframe, uv_H, uv_W), np.uint16)

    bytes2num = partial(int.from_bytes, byteorder='little', signed=False)

    bytesPerPixel = math.ceil(bitdepth / 8)
    seekPixels = startframe * H * W * 3 // 2
    fp = open(filename, 'rb')
    fp.seek(bytesPerPixel * seekPixels)

    for i in range(totalframe):
        for m in range(H):
            for n in range(W):
                if bitdepth == 8:
                    pel = bytes2num(fp.read(1))
                    Y[i, m, n] = np.uint8(pel)
                elif bitdepth == 10:
                    pel = bytes2num(fp.read(2))
                    Y[i, m, n] = np.uint16(pel)

        for m in range(uv_H):
            for n in range(uv_W):
                if bitdepth == 8:
                    pel = bytes2num(fp.read(1))
                    U[i, m, n] = np.uint8(pel)
                elif bitdepth == 10:
                    pel = bytes2num(fp.read(2))
                    U[i, m, n] = np.uint16(pel)

        for m in range(uv_H):
            for n in range(uv_W):
                if bitdepth == 8:
                    pel = bytes2num(fp.read(1))
                    V[i, m, n] = np.uint8(pel)
                elif bitdepth == 10:
                    pel = bytes2num(fp.read(2))
                    V[i, m, n] = np.uint16(pel)
    return Y, U, V


def writeyuv420_single(Y, U, V, filename):
    #BasketballDrill_832x480_50fps_8bit_420_00.yuv
    #0.002_BasketballDrill_832x480_50fps_8bit_420_00.yuv
    file_basename = os.path.basename(filename)
    if '8bit' in file_basename:
        bitdepth = 8
    elif '10bit' in file_basename:
        bitdepth = 10
    if len(np.shape(Y)) == 3:
        totalframe, H, W = np.shape(Y)
    elif len(np.shape(Y)) == 4:
        totalframe, H, W, _ = np.shape(Y)
    uv_H = np.shape(U)[1]
    uv_W = np.shape(U)[2]
    fp = open(filename, 'wb')

    for i in range(totalframe):
        for m in range(H):
            for n in range(W):
                if bitdepth == 8:
                    pel = bytes(np.array([Y[i, m, n]], dtype=np.uint8))
                    fp.write(pel)
                elif bitdepth == 10:
                    pel = bytes(np.array([Y[i, m, n]], dtype=np.uint16))
                    fp.write(pel)

        for m in range(uv_H):
            for n in range(uv_W):
                if bitdepth == 8:
                    pel = bytes(np.array([U[i, m, n]], dtype=np.uint8))
                    fp.write(pel)
                elif bitdepth == 10:
                    pel = bytes(np.array([U[i, m, n]], dtype=np.uint16))
                    fp.write(pel)

        for m in range(uv_H):
            for n in range(uv_W):
                if bitdepth == 8:
                    pel = bytes(np.array([V[i, m, n]], dtype=np.uint8))
                    fp.write(pel)
                elif bitdepth == 10:
                    pel = bytes(np.array([V[i, m, n]], dtype=np.uint16))
                    fp.write(pel)


def readyuv420_single_fast(filename, startframe=0, totalframe=1):
    #BasketballDrill_832x480_50fps_8bit_420_00.yuv
    #0.002_BasketballDrill_832x480_50fps_8bit_420_00.yuv
    file_basename = os.path.basename(filename)
    Frame_size = re.findall('_(\d+?)x(\d+?)_', file_basename)
    W = int(Frame_size[0][0])
    H = int(Frame_size[0][1])

    uv_H = H // 2
    uv_W = W // 2

    if '8bit' in file_basename:
        bitdepth = 8
        stream = np.fromfile(filename, np.uint8)
    elif '10bit' in file_basename:
        bitdepth = 10
        stream = np.fromfile(filename, np.uint16)
    Y = np.reshape(stream[0:H * W], [totalframe, H, W])
    U = np.reshape(stream[H * W:H * W + uv_H * uv_W], [totalframe, uv_H, uv_W])
    V = np.reshape(stream[H * W + uv_H * uv_W:H * W + uv_H * uv_W * 2], [totalframe, uv_H, uv_W])
    return Y.squeeze(), U.squeeze(), V.squeeze()


def readyuv420_dataset(filename, size=160, startframe=0, totalframe=1):
    #BasketballDrill_832x480_50fps_8bit_420_00.yuv
    #0.002_BasketballDrill_832x480_50fps_8bit_420_00.yuv
    file_basename = os.path.basename(filename)
    Frame_size = re.findall('_(\d+?)x(\d+?)_', file_basename)
    W = int(Frame_size[0][0])
    H = int(Frame_size[0][1])

    uv_H = H // 2
    uv_W = W // 2

    if '8bit' in file_basename:
        bitdepth = 8
        stream = np.fromfile(filename, np.uint8)
    elif '10bit' in file_basename:
        bitdepth = 10
        stream = np.fromfile(filename, np.uint16)
    Y = np.reshape(stream[0:H * W], [totalframe, H, W])
    U = np.reshape(stream[H * W:H * W + uv_H * uv_W], [totalframe, uv_H, uv_W])
    V = np.reshape(stream[H * W + uv_H * uv_W:H * W + uv_H * uv_W * 2], [totalframe, uv_H, uv_W])
    UV = np.concatenate((U, V), axis=1)
    return np.concatenate((Y, UV), axis=2)


def readyuv444_dataset(filename, size=160, startframe=0, totalframe=1):
    #BasketballDrill_832x480_50fps_8bit_420_00.yuv
    #0.002_BasketballDrill_832x480_50fps_8bit_420_00.yuv
    file_basename = os.path.basename(filename)
    Frame_size = re.findall('_(\d+?)x(\d+?)_', file_basename)
    W = int(Frame_size[0][0])
    H = int(Frame_size[0][1])
    uv_H = H
    uv_W = W

    if '8bit' in file_basename:
        stream = np.fromfile(filename, np.uint8)
    elif '10bit' in file_basename:
        stream = np.fromfile(filename, np.uint16)
    Y = np.reshape(stream[0:H * W], [H, W, 1])
    U = np.reshape(stream[H * W:H * W + uv_H * uv_W], [uv_H, uv_W, 1])
    V = np.reshape(stream[H * W + uv_H * uv_W:H * W + uv_H * uv_W * 2], [uv_H, uv_W, 1])
    return np.concatenate((Y, U, V), axis=2)


def writeyuv420_single_fast(Y, U, V, filename):
    #BasketballDrill_832x480_50fps_8bit_420_00.yuv
    #0.002_BasketballDrill_832x480_50fps_8bit_420_00.yuv
    file_basename = os.path.basename(filename)
    if '8bit' in file_basename:
        bitdepth = 8
    elif '10bit' in file_basename:
        bitdepth = 10
    if len(np.shape(Y)) == 3:
        totalframe, H, W = np.shape(Y)
    elif len(np.shape(Y)) == 4:
        totalframe, H, W, _ = np.shape(Y)
    uv_H = np.shape(U)[1]
    uv_W = np.shape(U)[2]
    Y = np.reshape(Y, [H * W])
    U = np.reshape(U, [uv_H * uv_W])
    V = np.reshape(V, [uv_H * uv_W])
    stream = np.concatenate((Y, U, V))
    stream.tofile(filename)


def readyuv444_single_fast(filename, startframe=0, totalframe=1):
    #BasketballDrill_832x480_50fps_8bit_420_00.yuv
    #0.002_BasketballDrill_832x480_50fps_8bit_420_00.yuv
    file_basename = os.path.basename(filename)
    Frame_size = re.findall('_(\d+?)x(\d+?)_', file_basename)
    W = int(Frame_size[0][0])
    H = int(Frame_size[0][1])

    uv_H = H
    uv_W = W

    if '8bit' in file_basename:
        stream = np.fromfile(filename, np.uint8)
    elif '10bit' in file_basename:
        stream = np.fromfile(filename, np.uint16)
    Y = np.reshape(stream[0:H * W], [totalframe, H, W])
    U = np.reshape(stream[H * W:H * W + uv_H * uv_W], [totalframe, uv_H, uv_W])
    V = np.reshape(stream[H * W + uv_H * uv_W:H * W + uv_H * uv_W * 2], [totalframe, uv_H, uv_W])
    return Y, U, V

def readyuv444_from_pkl(filename, startframe=0, totalframe=1):
    # BasketballDrill_832x480_50fps_8bit_420_00.yuv
    # 0.002_BasketballDrill_832x480_50fps_8bit_420_00.yuv
    with open(filename, 'rb') as f:
        import pickle
        rgb = pickle.load(f)
        rgb = np.array(rgb).astype(np.float32) / 255.
        yuv = rgb2yuv(rgb.transpose(2,0,1))
        yuv = (yuv * 255 + 0.5).clip(0., 255.)
        yuv = np.floor(yuv).astype(np.uint8)

    Y = yuv[0,:,:]
    U = yuv[1,:,:]
    V = yuv[2,:,:]
    h, w = Y.shape
    h = (h//2) * 2
    w = (w//2) * 2
    return Y[0:h, 0:w], U[0:h, 0:w], V[0:h, 0:w]

def readyuv420_from_pkl(filename, startframe=0, totalframe=1):
    # BasketballDrill_832x480_50fps_8bit_420_00.yuv
    # 0.002_BasketballDrill_832x480_50fps_8bit_420_00.yuv
    with open(filename, 'rb') as f:
        import pickle
        yuv = pickle.load(f)
    Y = yuv['y']
    U = yuv['u']
    V = yuv['v']
    h, w = Y.shape
    h = (h//2) * 2
    w = (w//2) * 2
    return Y[0:h, 0:w], U[0:h//2, 0:w//2], V[0:h//2, 0:w//2]

def writeyuv444_single_fast(Y, U, V, filename):
    #BasketballDrill_832x480_50fps_8bit_420_00.yuv
    #0.002_BasketballDrill_832x480_50fps_8bit_420_00.yuv
    if len(np.shape(Y)) == 3:
        _, H, W = np.shape(Y)
    elif len(np.shape(Y)) == 4:
        _, H, W, _ = np.shape(Y)
    uv_H = np.shape(U)[1]
    uv_W = np.shape(U)[2]
    Y = np.reshape(Y, [H * W])
    U = np.reshape(U, [uv_H * uv_W])
    V = np.reshape(V, [uv_H * uv_W])
    stream = np.concatenate((Y, U, V))
    stream.tofile(filename)


if __name__ == '__main__':
    y, u, v = readyuv420_single_fast(
        r'../data/val/huawei_all_test_yuv/ori/BasketballDrill_832x480_50fps_8bit_420_00.yuv')
    writeyuv420_single_fast(y, u, v, r'./BasketballDrill_832x480_50fps_8bit_420_00.yuv')
    y, u, v = readyuv420_single_fast(
        r'../data/val/huawei_all_test_yuv/ori/CatRobot_3840x2160_60fps_10bit_420_00.yuv')
    writeyuv420_single_fast(y, u, v, r'./CatRobot_3840x2160_60fps_10bit_420_00.yuv')
