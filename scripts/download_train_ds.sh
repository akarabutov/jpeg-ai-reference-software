#!/bin/bash

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
