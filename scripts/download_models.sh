#!/bin/bash

env_name="jpeg_ai_vm"
if (( $# > 0 )); then
    env_name=$1
fi

eval "$(conda shell.bash hook)"

cur_dir=`pwd`

conda activate ${env_name}

dvc pull ${cur_dir}/models/*/*.dvc