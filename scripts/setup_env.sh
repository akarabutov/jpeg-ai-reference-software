#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

env_name="jpeg_ai_vm"
if (( $# > 0 )); then
    env_name=$1
fi

eval "$(conda shell.bash hook)"
cur_dir=`pwd`

conda create -n ${env_name} python=3.7 -y
conda activate ${env_name}
pip install -r requirements.txt
pre-commit install
${SCRIPT_DIR}/build_test_libs.sh
sshpass -p kj3457rjzsd sftp storage_reader@jpeg-git.lx.it.pt << !
    bye
!
