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

# go to line 58 to change which qps are processed.

import argparse
import glob
import random
import time
import os
import numpy as np
import re
from yuv2img_simple_nn import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_dir_label', dest='src_dir', default='../../dataset/yuv/bop_org', help='dir of ground truth data')
parser.add_argument('--src_dir_bayer', dest='src_dir_bayer', default='../../dataset/yuv/bop_', help='dir of interpolated Bayer data')

parser.add_argument('--save_dir_label', dest='save_dir_label', default='../../dataset/subimages/bop_org', help='dir of patches')
parser.add_argument('--save_dir_bayer', dest='save_dir_bayer', default='../../dataset/subimages/bop_', help='dir of patches')
parser.add_argument('--patch_size', dest='pat_size', type=int, default=192, help='patch size')
parser.add_argument('--stride', dest='stride', type=int, default=180, help='stride')
parser.add_argument('--step', dest='step', type=int, default=0, help='step, not starting from beginning')
parser.add_argument('--batch_size', dest='bat_size', type=int, default=64, help='Planned batch size')
parser.add_argument('--debug', dest='isDebug', type=bool, default=False, help='Debug Mode, using a few for debug')
parser.add_argument('--joint', dest='MultiQP', type=bool, default=False, help='Multi QP mode, blind generation')
args = parser.parse_args()

def generate_patches():
    isDebug = args.isDebug
    filepaths = sorted(glob.glob(args.src_dir + '/*.yuv'))
    if isDebug:
        numDebug = 10
        filepaths = filepaths[:numDebug] # take only ten images to quickly debug
    print("number of training images %d" % len(filepaths))
    count1 = 0 # calculate the number of patches
    for i in range(len(filepaths)):
        file_basename = os.path.basename(filepaths[i])
        #print(file_basename)
        Frame_size = re.findall("_(\d+?)x(\d+?)(_|.yuv$)", file_basename)
        im_w = int(Frame_size[0][0])
        im_h = int(Frame_size[0][1])
        #print(im_w)
        #print(im_h)
        im_w = im_h = 1024
        if int(Frame_size[0][0]) <= 1024 or int(Frame_size[0][1]) <= 1024:
            im_w = (int(Frame_size[0][0]) // 2) * 2
            im_h = (int(Frame_size[0][1]) // 2) * 2

        for x in range(0 + args.step, (im_h - args.pat_size), args.stride):
            for y in range(0 + args.step, (im_w - args.pat_size), args.stride):
                count1 += 1

    origin_patch_num = count1
    if origin_patch_num % args.bat_size != 0:
        numPatches = int(origin_patch_num / args.bat_size) * args.bat_size
    else:
        numPatches = int(origin_patch_num)
    print("Total patches = %d , batch size = %d, total batches = %d" %(numPatches, args.bat_size, numPatches / args.bat_size))
    time.sleep(5)
    
    random.seed(4)
    random.shuffle(filepaths)
    # generate patches
    img_idx = 0
    for i in range(len(filepaths)):
        if not args.MultiQP:
            img_idx = img_idx + 1
        for qp in [12, 25, 50, 75, 100]: #12, 25, 50, 75, 100
            if not args.MultiQP:
                gt_path  = args.save_dir_label
                vvc_path = args.save_dir_bayer + '%03d' %(qp)
            else:
                img_idx = img_idx + 1
                gt_path  = args.save_dir_label + '_ALL'
                vvc_path = args.save_dir_bayer + '_ALL'
            if not os.path.exists(gt_path):
                os.mkdir(gt_path)
            if not os.path.exists(vvc_path):
                os.mkdir(vvc_path)
            filepaths_bayer = sorted(glob.glob(args.src_dir_bayer + '%03d/*' %(qp)))
            if isDebug:
                filepaths_bayer = filepaths_bayer[:numDebug]
            random.seed(4)
            random.shuffle(filepaths_bayer)
            img_Y, img_U, img_V = readyuv444_single_fast(filepaths[i])
            img_Bayer_Y, img_Bayer_U, img_Bayer_V = readyuv444_single_fast(filepaths_bayer[i])
            if '8bit' in os.path.basename(filepaths[i]):
                bit_depth = 8
            elif '10bit' in os.path.basename(filepaths[i]):
                bit_depth = 10
            else:
                print('ERROR!!! No bit depth information')
                break

            gt_name = os.path.basename(filepaths[i])
            gt_name = gt_name.replace(os.path.basename(filepaths[i]).split('_')[0], '{:05d}'.format(img_idx)) #Rename with an index

            img_Y       = np.squeeze(img_Y)
            img_U       = np.squeeze(img_U)
            img_V       = np.squeeze(img_V)
            img_Bayer_Y = np.squeeze(img_Bayer_Y)
            img_Bayer_U = np.squeeze(img_Bayer_U)
            img_Bayer_V = np.squeeze(img_Bayer_V)
            im_h, im_w  = img_Y.shape
            print("The %dth image of %d training images of QP %03d, bit depth %d bit" %(i+1, len(filepaths), qp, bit_depth))
            count = 0
            for x in range(0 + args.step, im_h - args.pat_size, args.stride):
                for y in range(0 + args.step, im_w - args.pat_size, args.stride):
                    image_label_Y = img_Y[x:x + args.pat_size, y:y + args.pat_size]
                    image_label_U = img_U[x:x + args.pat_size, y:y + args.pat_size]
                    image_label_V = img_V[x:x + args.pat_size, y:y + args.pat_size]
                    
                    image_bayer_Y = img_Bayer_Y[x:x + args.pat_size, y:y + args.pat_size]
                    image_bayer_U = img_Bayer_U[x:x + args.pat_size, y:y + args.pat_size]
                    image_bayer_V = img_Bayer_V[x:x + args.pat_size, y:y + args.pat_size]
                    image_label_Y = np.expand_dims(image_label_Y, 0)
                    image_label_U = np.expand_dims(image_label_U, 0)
                    image_label_V = np.expand_dims(image_label_V, 0)
                    image_bayer_Y = np.expand_dims(image_bayer_Y, 0)
                    image_bayer_U = np.expand_dims(image_bayer_U, 0)
                    image_bayer_V = np.expand_dims(image_bayer_V, 0)
                    target_name = gt_name.replace('.yuv', '_s{:03d}.yuv'.format(count))
                    target_name = target_name.replace(target_name.split('_')[1], str(args.pat_size)+'x'+str(args.pat_size))
                    writeyuv444_single_fast(image_label_Y, image_label_U, image_label_V, os.path.join(gt_path, target_name))
                    writeyuv444_single_fast(image_bayer_Y, image_bayer_U, image_bayer_V, os.path.join(vvc_path, target_name))
                    count = count + 1
    print("Total patches = %d , batch size = %d, total batches = %d" %(numPatches, args.bat_size, numPatches / args.bat_size))
    print("Training data has been written into TFrecord.")

if __name__ == '__main__':
    generate_patches()
