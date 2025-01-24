data_dir=<folder path of original training images>
# The copyright in this software is being made available under the BSD
#  License, included below. This software may be subject to other third party
#  and contributor rights, including patent rights, and no such rights are
#  granted under this license.
# 
#  Copyright (c) 2010-2022, ITU/ISO/IEC
#  All rights reserved.
# 
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
# 
#  * Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
#  be used to endorse or promote products derived from this software without
#  specific prior written permission.
# 
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
#  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
#  THE POSSIBILITY OF SUCH DAMAGE.

lst=<path of image list>

echo mode-I: random cropping
python image_crop.py --lst ${lst} \
                     --data_dir ${data_dir} \
                     --crop_size 1024 \
                     --crop_format random \
                     --save_data_dir <output path of images> \
                     --output_info <output path of image info> \
                     --output_lst <output path of image list> \

#echo mode-II: cropping with sliding window
#python image_crop.py --lst ${lst} \
#                     --data_dir ${data_dir} \
#                     --crop_size 1024 \
#                     --crop_format sliding \
#                     --save_data_dir <output path of images> \
#                     --output_info <output path of image info> \
#                     --output_lst <output path of image list> \
