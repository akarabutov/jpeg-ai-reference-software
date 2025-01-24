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


#
# Run the script from root of the project.
# Command line: ./scripts/run_reco_all_cfgs.sh <BASE_DIRECTORY_PATH> [<GPU_EVAL_CFG>]
#
# The script will run simulation and put them to <BASE_DIRECTORY_PATH>/<oper_point>/tools_on-GPU, <BASE_DIRECTORY_PATH>/<oper_point>/tools_off-GPU and etc.
#

additional_arg="--resume_eval"
# 

CPU_THREADS_LIMIT=8
RUN_CPU_SIMS=0


CUR_PWD=`pwd`
OUTPUT_BASE_DIR="${CUR_PWD}/results/`git rev-parse --abbrev-ref HEAD`___`git rev-parse --short HEAD`"
if (( $# > 0 )); then
    OUTPUT_BASE_DIR=$1
fi

GPU_EVAL_CFG="./cfg/eval/tools_onoff.json"
if (( $# > 1 )); then
    GPU_EVAL_CFG=$2
fi

if (( $# > 2 )); then
    additional_arg="$additional_arg ${@:3}"
fi


TEST_NAME=`basename ${OUTPUT_BASE_DIR}`

# GPU version for all tasks
python ${CUR_PWD}/scripts/run_eval_script.py ${OUTPUT_BASE_DIR} --cfg ${GPU_EVAL_CFG} ${additional_arg} 
python ${CUR_PWD}/scripts/merge_op_results.py ${OUTPUT_BASE_DIR} --fn-prefix ${TEST_NAME}_
# Generate excel files with results
for oper_point_dir in ${OUTPUT_BASE_DIR}/*
do
    if [ -d "${oper_point_dir}" ]
    then
        oper_point=`basename ${oper_point_dir}`
        echo "Merging of all summaries in ${oper_point_dir}"
        python ./scripts/merge_summaries.py ${oper_point_dir} --prefix ${TEST_NAME} --fn-prefix ${TEST_NAME}_${oper_point}
    fi
done

if [ $RUN_CPU_SIMS == 1 ]
then
    # CPU version
    for oper_point_dir in ${OUTPUT_BASE_DIR}/*
    do
        if [ -d "${oper_point_dir}" ]
        then
            oper_point=`basename ${oper_point_dir}`    
            additional_cfgs="cfg/oper_point/${oper_point}.json"
            OUTPUT_BASE_DIR_CUR="${OUTPUT_BASE_DIR}/${oper_point}"
            mkdir -p ${OUTPUT_BASE_DIR_CUR}
            OUTPUT_BASE_DIR_CUR=`realpath $OUTPUT_BASE_DIR_CUR`

            cfg_set=("tools_off")
            #cfg_set=("tools_off" "tools_on")
            for cfg in "${cfg_set[@]}"
            do
                CUDA_VISIBLE_DEVICES="-1" python -m src.reco.scripts.eval --coding_type enc_dec --cfg ./cfg/${cfg}.json ${additional_cfgs} --out_dir ${OUTPUT_BASE_DIR_CUR}/${cfg}-CPU --only_cpu ${additional_arg} --cpu_threads_limit ${CPU_THREADS_LIMIT} 2>&1 | tee ${OUTPUT_BASE_DIR_CUR}/${cfg}-CPU.log
            done

            # Checking CPU/GPU interoperability
            for cfg in "${cfg_set[@]}"
            do
                # Create new directory and link GPU results
                mkdir -p ${OUTPUT_BASE_DIR_CUR}/${cfg}-encGPU-decCPU
                ln -s ${OUTPUT_BASE_DIR_CUR}/${cfg}-GPU/bit ${OUTPUT_BASE_DIR_CUR}/${cfg}-encGPU-decCPU/
                mkdir -p ${OUTPUT_BASE_DIR_CUR}/${cfg}-encGPU-decCPU/log
                ln -s ${OUTPUT_BASE_DIR_CUR}/${cfg}-GPU/log/enc ${OUTPUT_BASE_DIR_CUR}/${cfg}-encGPU-decCPU/log/
                # Run CPU decoder with checking hashes of reconstructed images and metrics calculation.
                CUDA_VISIBLE_DEVICES="-1" python -m src.reco.scripts.eval --coding_type dec --cfg ./cfg/${cfg}.json ${additional_cfgs} --out_dir ${OUTPUT_BASE_DIR_CUR}/${cfg}-encGPU-decCPU --only_cpu --force_encdec_match 1 --calc_decoder_metrics 1 ${additional_arg} --cpu_threads_limit ${CPU_THREADS_LIMIT}  2>&1 | tee ${OUTPUT_BASE_DIR_CUR}/${cfg}-encGPU-decCPU.log

                # Create new directory and link CPU results
                mkdir -p ${OUTPUT_BASE_DIR_CUR}/${cfg}-encCPU-decGPU
                ln -s ${OUTPUT_BASE_DIR_CUR}/${cfg}-CPU/bit ${OUTPUT_BASE_DIR_CUR}/${cfg}-encCPU-decGPU/
                mkdir -p ${OUTPUT_BASE_DIR_CUR}/${cfg}-encCPU-decGPU/log
                ln -s ${OUTPUT_BASE_DIR_CUR}/${cfg}-CPU/log/enc ${OUTPUT_BASE_DIR_CUR}/${cfg}-encCPU-decGPU/log/
                # Run GPU decoder with checking hashes of reconstructed images and metrics calculation.
                python -m src.reco.scripts.eval --coding_type dec --cfg ./cfg/${cfg}.json ${additional_cfgs} --out_dir ${OUTPUT_BASE_DIR_CUR}/${cfg}-encCPU-decGPU --force_encdec_match 1 --calc_decoder_metrics 1 ${additional_arg} 2>&1 | tee ${OUTPUT_BASE_DIR_CUR}/${cfg}-encCPU-decGPU.log
            done
            echo "Merging of all summaries in ${OUTPUT_BASE_DIR_CUR}"
            python ./scripts/merge_summaries.py ${OUTPUT_BASE_DIR_CUR} --prefix ${TEST_NAME} --fn-prefix ${TEST_NAME}_${oper_point}
        fi
    done
fi