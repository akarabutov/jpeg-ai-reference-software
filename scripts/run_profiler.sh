#!/bin/bash

additional_arg="--imgs 00038_TE_8160x6120_8bit_sRGB.png -Profilers.module_duration 1"

CUR_PWD=`pwd`
OUTPUT_BASE_DIR="${CUR_PWD}/profiler/`git rev-parse --abbrev-ref HEAD`___`git rev-parse --short HEAD`"
if (( $# > 0 )); then
    OUTPUT_BASE_DIR=$1
fi

GPU_EVAL_CFG="./cfg/eval/tools_off.json"
if (( $# > 1 )); then
    GPU_EVAL_CFG=$2
fi


# GPU version for all tasks
python ${CUR_PWD}/scripts/run_eval_script.py ${OUTPUT_BASE_DIR} --cfg ${GPU_EVAL_CFG} ${additional_arg}  --profiler_path ${OUTPUT_BASE_DIR}