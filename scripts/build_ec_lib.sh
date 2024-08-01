#!/bin/bash

env_name="jpeg_ai_vm"
if (( $# > 0 )); then
    env_name=$1
fi

eval "$(conda shell.bash hook)"

cur_dir=`pwd`

conda activate ${env_name}

# me-tANS
cd ${cur_dir}/src/codec/entropy_coding/cpp_exts/mans && make
# engine to work with headers
cd ${cur_dir}/src/codec/entropy_coding/cpp_exts/direct && make