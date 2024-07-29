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
import math
import os
import random
import time

import cv2

parser = argparse.ArgumentParser(description='The crop of the JPEG-AI training Images.')
# parameters of dataset
parser.add_argument('--lst', type=str, default='', help='')
parser.add_argument('--data_dir', type=str, default='', help='')
parser.add_argument('--save_data_dir', type=str, default='', help='')
parser.add_argument('--output_lst', type=str, default='', help='')
parser.add_argument('--output_info', type=str, default='', help='')
# croping parameters
parser.add_argument('--crop_size', type=int, default=1024, help='')
parser.add_argument('--seed', type=int, default=123, help='')
parser.add_argument('--crop_format',
                    type=str,
                    choices=['sliding', 'random'],
                    default='random',
                    help='')

args, unparsed = parser.parse_known_args()


def sliding_window(iw, ih, crop_size):
    coordinates = []
    count_w = math.ceil(iw / crop_size)
    count_h = math.ceil(ih / crop_size)
    for i in range(count_w):
        for j in range(count_h):
            x = i * crop_size if i < count_w - 1 else iw - crop_size
            y = j * crop_size if j < count_h - 1 else ih - crop_size
            if [x, y] not in coordinates:
                coordinates.append([x, y])
    return coordinates


def random_crop(iw, ih, crop_size):
    coordinates = []
    count_w = math.ceil(iw / crop_size)
    count_h = math.ceil(ih / crop_size)
    crop_number = count_w * count_h * 2
    for n in range(crop_number):
        # obtain the random parameter
        x = random.randint(0, iw - crop_size)
        y = random.randint(0, ih - crop_size)
        coordinates.append([x, y])

    return coordinates


def main():
    """The main function.
    """
    if args.seed is not None:
        random.seed(args.seed)

    if not os.path.exists(args.save_data_dir):
        os.makedirs(args.save_data_dir)
    #obtain crop size
    th, tw = args.crop_size, args.crop_size

    f_info = open(args.output_info, 'a')
    f_lst = open(args.output_lst, 'a')

    with open(args.lst, 'r') as f_in_lst:
        lines = f_in_lst.readlines()
    count = 0
    for line in lines:
        image_name = line.strip().split(' ')[0]
        abs_path = os.path.join(args.data_dir, image_name)
        if not os.path.exists(abs_path):
            print('The image not exists', abs_path)
            continue
        image = cv2.imread(abs_path)
        w, h, _ = image.shape
        if w <= args.crop_size or h <= args.crop_size:
            save_path = os.path.join(args.save_data_dir, image_name)
            cv2.imwrite(save_path, image)
            image_info = '{} {} {} {} {}'.format(image_name, 0, w, 0, h)
            print(image_info, 'has been saved')
            f_info.write(image_info + '\n')
            f_lst.write(image_name + '\n')
            count += 1
            continue
        if args.crop_format == 'random':
            coordinates = random_crop(w, h, args.crop_size)
        else:
            coordinates = sliding_window(w, h, args.crop_size)

        for n in range(len(coordinates)):
            j, i = coordinates[n][0], coordinates[n][1]
            # crop the image
            image_crop = image[j:j + tw, i:i + th, :]
            #save the cropped image
            image_name_new = image_name[:-4] + '_' + str(n) + image_name[-4:]
            save_path = os.path.join(args.save_data_dir, image_name_new)
            cv2.imwrite(save_path, image_crop)
            image_info = '{} {} {} {} {}'.format(image_name_new, i, i + th, j, j + tw)
            print(image_info, 'has been saved')
            f_info.write(image_info + '\n')
            f_lst.write(image_name_new + '\n')
            count += 1
    f_info.write('The total number of images is ' + str(count))
    f_info.close()
    f_lst.close()


if __name__ == '__main__':
    main()
