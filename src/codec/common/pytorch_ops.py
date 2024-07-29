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

import torch
import torch.nn.functional as F
import hashlib


########################################################################################################################
# class TorchOps
########################################################################################################################
def disable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def disable_torch_random():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def enable_torch_random():
    # benchmark = True is not recommended when AMP is using in training.
    # It does not speed up the training.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def determinism(func):
    def wrapper(self, *args, **kwargs):
        disable_torch_random()
        disable_tf32()
        val = func(self, *args, **kwargs)
        return val

    return wrapper


def determinism_on_eval(func):
    def wrapper(self, *args, **kwargs):
        if not self.training:
            disable_torch_random()
            disable_tf32()
        else:
            enable_torch_random()
        val = func(self, *args, **kwargs)
        return val

    return wrapper


class TorchOps:
    """Torch Operations
    """
    @staticmethod
    def torch_version():
        print('Torch:', torch.__version__)

    @staticmethod
    def cuda_version():
        print('Cuda:', torch.version.cuda)

    @staticmethod
    def cudnn_version():
        print('Cudnn:', torch.backends.cudnn.version())

    @staticmethod
    def check_gpu():
        if torch.cuda.is_available():
            print('cuda is available!')
            return True
        else:
            print('cuda is unavailable!')
            return False

    @staticmethod
    def get_device_count():
        count = torch.cuda.device_count()
        print('{} GPUs can be used.'.format(count))
        return count

    @staticmethod
    def get_device(gpu_id=0, use_gpu=True):
        device = 'cpu'
        if use_gpu:
            count = TorchOps.get_device_count()
            if gpu_id < count:
                device = 'cuda:' + str(id)
            else:
                raise AssertionError('Invalid gpu_id={}'.format(gpu_id))
        return torch.device(device)

    @staticmethod
    def get_devices(gpu_ids, use_gpu=True):
        devices = list()

        if use_gpu:
            count = TorchOps.get_device_count()
            total = min(len(gpu_ids), count)
            for gpu_id in range(total):
                device = 'cuda:' + str(gpu_id)
                devices.append(torch.device(device))
        else:
            device = 'cpu'
            devices.append(torch.device(device))

        return devices


########################################################################################################################
# class TensorOps
########################################################################################################################
class TensorOps:
    """Tensor operations
    """
    @staticmethod
    def to_array(tensor):
        if not torch.is_tensor(tensor):
            raise TypeError('Input type must be torch.Tensor but with type={}'.format(
                type(tensor)))

        detach = tensor.detach()  # stop gradient
        array = detach.cpu().numpy()  # cuda tensor => cpu tensor => cpu numpy
        return array

    @staticmethod
    def to_dtype(type_name: str = None):
        if type_name == 'Tensor':
            dtype = torch.Tensor
        elif type_name == 'FloatTensor':
            dtype = torch.FloatTensor
        elif type_name == 'CudaFloatTensor' and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            raise AssertionError('Unsupported type_name={}'.format(type_name))
        return dtype

    @staticmethod
    def to_onehot(tensor, depth):
        """One hot transform.
        Args:
            tensor (tensor): the tensor to be transformed
            depth (int): the class number in one hot
        Returns:
            a new tensor with one hot value
        """
        shape = list(tensor.shape)
        shape.append(depth)
        onehot = torch.zeros(shape).to(tensor.device)
        tensor = tensor.unsqueeze(dim=-1)
        return onehot.scatter(-1, tensor, 1)

    @staticmethod
    def expand_dim_num(tensor: torch.Tensor, target_dim_num: int) -> torch.Tensor:
        """Expand number of dimentions by adding addtional

        Args:
            tensor (torch.Tensor): input tensor
            target_dim_num (int): desirable number of dimentionals

        Returns:
            torch.Tensor: output tensor with `target_dim_num` number of dimentions
        """
        assert target_dim_num >= tensor.ndim
        while target_dim_num > tensor.ndim:
            tensor.unsqueeze_(0)
        return tensor

    @staticmethod
    def resize_tensor(tensor: torch.Tensor,
                      new_size: torch.Size,
                      mode: str = 'bilinear',
                      align_corners: bool = True) -> torch.Tensor:
        return F.interpolate(tensor, size=new_size, mode=mode, align_corners=align_corners)

    @staticmethod
    def get_hash(tensor: torch.Tensor) -> str:
        return hashlib.md5(tensor.detach().contiguous().cpu().numpy()).hexdigest()

