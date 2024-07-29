#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/common.sh

OUTPUT_PATH=${SCRIPT_DIR}/../../.dvc/config.local
if (( $# > 0 )); then
    OUTPUT_PATH=$1
fi

cp ${SCRIPT_DIR}/config.tmp ${OUTPUT_PATH}

read -p "Enter sftp server's IP: " sftp_ip
read -p "Enter sftp user's password: " password

sed -i "s/{user}/$sftp_user/" ${OUTPUT_PATH}
sed -i "s/{password}/$password/" ${OUTPUT_PATH}
sed -i "s/{ip}/$sftp_ip/" ${OUTPUT_PATH}
