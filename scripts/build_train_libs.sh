#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

env_name="jpeg_ai_vm"
if (( $# > 0 )); then
    env_name=$1
fi

chmod 744 ${SCRIPT_DIR}/build_ec_lib.sh
${SCRIPT_DIR}/build_ec_lib.sh ${env_name}

eval "$(conda shell.bash hook)"

cur_dir=`pwd`

conda activate ${env_name}

cd src/train/3rdparty/apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option=build_ext --global-option="-I/usr/local/cuda/include/" ./
