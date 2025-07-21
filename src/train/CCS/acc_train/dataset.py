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
import random

import torch
from PIL import Image
from torch.utils.data import Dataset


class CodecDataset(Dataset):
    """Codec Dataset."""
    def __init__(self, root, lst, transform=None):
        """
        Args:
            lst (string): Path list.
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.lst = lst
        self.root = root
        self.transform = transform
        self.sequence = []
        with open(self.lst, 'r') as f:
            lines = f.readlines()
        for line in lines:
            self.sequence.append(line.strip().split(' ')[0])

    def __getitem__(self, index):
        """Transform per element to image.
        """
        path = self.sequence[index]
        abs_path = os.path.join(self.root, path)
        image = Image.open(abs_path).convert('RGB')  # convert fixed by Burak
        if self.transform:
            image = self.transform(image)
        return path, image

    def __len__(self):
        """Get the length of the dataset.
        """
        return len(self.sequence)


class CustomToTensor(object):
    """Convert a ``PIL Image`` to tensor.
    Converts a PIL Image (H x W x C) in the range [0, 255] to
    a torch.FloatTensor of shape (C x H x W) in the range [0.0, 255.0]
    """
    def __call__(self, pic):
        """
        Args:
            pic (PIL Image): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return custom_to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def custom_to_tensor(pic):
    """Convert a ``PIL Image`` to tensor. See ``ToTensor`` for more details.
    Args:
        pic (PIL Image): Image to be converted to tensor.
    Returns:
        Tensor: Converted image.
    """
    # handle PIL Image
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    nchannel = len(pic.split())
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float()


class CustomCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (int): Desired output size of the crop, a square crop (size, size) is made.
        num  (int): The number of regions to be croped from the raw image.
    """
    def __init__(self, size, num):
        self.size = (size, size)
        self.num = num

    @staticmethod
    def get_params(img, output_size, output_num):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img     (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
            output_num    (int): Expected number of regions.
        Returns:
            coordinates  (list): Store all the coordinates to be cropped.
        """
        w, h = img.size
        th, tw = output_size

        coordinates = []

        for _ in range(output_num):
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
            coordinates.append((i, j, th, tw))

        return coordinates

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            image_container (list): The list storing all the cropped images.
        """
        image_container = []

        for coordinate in self.get_params(img, self.size, self.num):
            image_container.append(custom_crop(img, *coordinate))

        return image_container

    def __repr__(self):
        return self.__class__.__name__ + '()'


def custom_crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (PIL Image): Image to be cropped.
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped image.
        w (int): Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    return img.crop((j, i, j + w, i + h))


class CustomResize(object):
    """Resize the input PIL Image to the given size.
    Args:
        threshold (sequence): Desired output size. =>(1024, 512, 256)
        interpolation (int, optional): Desired interpolation. Default is ``PIL.Image.BILINEAR``
    """
    def __init__(self, threshold, interpolation=Image.BILINEAR):
        self.threshold = threshold
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        w, h = img.size
        short = min(w, h)
        size = self.threshold[-1]
        for t in self.threshold:
            if short > t:
                size = t
                break
        return custom_resize(img, size, self.interpolation)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def custom_resize(img, size, interpolation=Image.BILINEAR):
    """Resize the input PIL Image to the given size.
    Args:
        img (PIL Image): Image to be resized.
        size (int): Desired output size.
        interpolation (int, optional): Desired interpolation. Default is ``PIL.Image.BILINEAR``
    Returns:
        PIL Image: Resized image.
    """
    w, h = img.size
    if (w <= h and w == size) or (h <= w and h == size):
        return img
    if w < h:
        ow = size
        oh = int(size * h / w)
        return img.resize((ow, oh), interpolation)
    else:
        oh = size
        ow = int(size * w / h)
        return img.resize((ow, oh), interpolation)
