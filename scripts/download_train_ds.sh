#!/bin/bash
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


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "You need to enter a password of sFTP. The password you can find here: https://sd.iso.org/documents/ui/#!/browse/iso/iso-iec-jtc-1/iso-iec-jtc-1-sc-29/iso-iec-jtc-1-sc-29-wg-1/library/6/98-Sydney/OUTPUT%20N-documents/wg1n100422-098-ICQ-Access%20information%20for%20JPEG%20AI%20dataset"
read -sp 'Password: ' passvar
USER=jpeg-ai
SFTP_ADDR=amalia.img.lx.it.pt

cd ${SCRIPT_DIR}/../data

echo
echo

sshpass -p $passvar sftp ${USER}@${SFTP_ADDR} << !
    mget /train_and_valid_natural/cropped/*.zip
    get /train_and_valid_scc700/scc7000_patchs2.tar
    bye
!

# Training dataset
unzip -j jpegai_training_random_crop_00000-01299.zip jpegai_training_random_crop_*/*.png -d jpegai_training_random_crop
unzip -j jpegai_training_random_crop_01300-02599.zip -d jpegai_training_random_crop
unzip -j jpegai_training_random_crop_02600-03899.zip jpegai_training_random_crop_*/*.png -d jpegai_training_random_crop
unzip -j jpegai_training_random_crop_03900-5263.zip jpegai_training_random_crop_*/*.png -d jpegai_training_random_crop
tar -tf scc7000_patchs2.tar -C jpegai_training_random_crop

# Generate a list of files in training dataset
ls -1 jpegai_training_random_crop/ > jpegai_training_random_crop/tmp.txt
sed '/\.txt/d' jpegai_training_random_crop/tmp.txt > jpegai_training_random_crop/jpegai_training_set512_random_crop_16.txt
rm jpegai_training_random_crop/tmp.txt

# Validation dataset
unzip -j jpegai_validation_set.zip -d jpegai_validation_set
# Generate a list of files in validation dataset
#ls -1 jpegai_validation_set/ > jpegai_validation_set/jpegai_validation_set_10.txt	# commented, because it is in the archive
