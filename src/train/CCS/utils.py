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
import shutil

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name='unknown'):
        self.reset()
        self.name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        if torch.is_tensor(val):
            val = val.detach().cpu().item()

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename='checkpoint.pth', is_best=False):
    """Save the checkpoint.
    """
    torch.save(state, filename)
    if is_best:
        dir_name = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(dir_name, 'best.pth'))


def weights_init(m):
    """Init the weights.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def to_one_hot(t, depth):
    """One hot transform.
    Args:
        t  (tensor): the tensor to be transformed
        depth (int): the class number in one hot label
    Returns:
        a new tensor with one hot value
    """
    one_hot_shape = list(t.shape)
    one_hot_shape.append(depth)
    t = t.unsqueeze(dim=-1)
    one_hot = torch.zeros(one_hot_shape).cuda()
    return one_hot.scatter(-1, t, 1)


def save_image(image_dir, image_name, image_data):
    """Save image with `png` format.
    Args:
        image_dir  (str): image dir to be saved
        image_name (str): image name to be saved
        image_data (tensor): the N3HW tensor
    """
    #image_name = image_name[0]
    assert image_name.endswith('.png')
    assert image_data.dim() == 4 and image_data.shape[0] == 1 and image_data.shape[1] == 3
    abs_image_name = os.path.join(image_dir, image_name)
    os.makedirs(os.path.dirname(abs_image_name), exist_ok=True)
    # float => uint8
    image_data = image_data.to(dtype=torch.uint8)
    # CUDA tensor => ndarray
    image_data = image_data.cpu().numpy()
    # NCHW => HWC
    image_data = np.transpose(image_data[0, :, :, :], (1, 2, 0))
    image = Image.fromarray(image_data)
    image.save(abs_image_name)


def save_yuv_image(image_dir, image_name, image_data):
    """Save image with `png` format.
    Args:
        image_dir  (str): image dir to be saved
        image_name (str): image name to be saved
        image_data (tensor): the N3HW tensor
    """
    #image_name = image_name[0]
    image_name = image_name[:-4] + '.yuv'
    assert image_name.endswith('.yuv')
    assert image_data.dim() == 4 and image_data.shape[0] == 1 and image_data.shape[1] == 3
    os.makedirs(image_dir, exist_ok=True)
    abs_image_name = os.path.join(image_dir, image_name)
    # float => uint8
    image_data = image_data.to(dtype=torch.uint8)
    y_part = image_data[0, 0, :, :].contiguous().view(-1)
    u_part = image_data[0, 1, :, :].contiguous().view(-1)
    v_part = image_data[0, 2, :, :].contiguous().view(-1)
    # CUDA tensor => ndarray
    y_part = y_part.cpu().numpy()
    u_part = u_part.cpu().numpy()
    v_part = v_part.cpu().numpy()
    yuv_arr = np.zeros(len(y_part) + len(u_part) + len(v_part), np.uint8)
    yuv_arr[:len(y_part)] = y_part
    yuv_arr[len(y_part):len(y_part) + len(u_part)] = u_part
    yuv_arr[len(y_part) + len(u_part):len(y_part) + len(u_part) + len(v_part)] = v_part
    # save to file
    yuv_arr.tofile(abs_image_name)


def mox_copy_with_timeout_retry(src_data, dst_data, retry_num, timeout, path, file_or_not):
    """Use this Func to substitude moxi.file.copy and moxi.file.copy_parallel.
    """
    status = 0
    # set the cmd string to excute
    cmd = 'timeout %(timeout)s python %(path)s %(src)s %(dst)s %(file_or_not)s' % {
        'timeout': timeout,
        'src': src_data,
        'dst': dst_data,
        'path': path,
        'file_or_not': file_or_not
    }
    print(cmd)
    for i in range(0, retry_num):
        ret = os.system(cmd)
        if ret == 0:
            print('copy success')
            status = 1
            break
        print('ret: %d retry' % (i + 1))
    if status != 1:
        print('copy fail exit')
        return 1
    else:
        return 0
