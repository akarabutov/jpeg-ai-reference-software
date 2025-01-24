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


DO_DATA_PREPARATION=1
DO_QUANTIZATION=1

WEIGHT_BD=8

CALIBRATION_SET=./data/calibration_set/

LATENTS_LIST_TO_DUMP="model_y.z_hat model_uv.z_hat model_y.residual_quant model_uv.residual_quant"

env_name="jpeg_ai_vm"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ${SCRIPT_DIR}/..

eval "$(conda shell.bash hook)"

conda activate ${env_name}
dvc pull data/calibration_set/*.dvc

LOCATION_OF_PREPARED_DATA=data/dump

if (( ${DO_DATA_PREPARATION} == 1 )); then
    for oper_point in "bop" "hop"
    do
        additional_cfgs="cfg/oper_point/${oper_point}.json"
        echo " === Dumping the intermediate data === "
        python -m src.dump.scripts.eval --coding_type enc --cfg cfg/tools_off.json ${additional_cfgs} cfg/AE/lh.json \
          --in_dir $CALIBRATION_SET --out_dir ${LOCATION_OF_PREPARED_DATA}/${oper_point} --latents_list $LATENTS_LIST_TO_DUMP --skip_models_check \
          -model.CCS_SGMM.tools_common.model_common.common_modules.ckpt_model_name VM_common \
          --calc_encoder_metrics 0
        echo " === Data preparation done === "
    done
fi

if (( ${DO_QUANTIZATION} == 1 )); then
    FLOAT_MODEL_PATH=models/VM_common
    RESULT_INT_MODEL_NAME=models/VM_common_int  
    
    echo " === Model quantization ==="
    CODEC_PARAMS="--cfg cfg/tools_off.json cfg/oper_point/bop.json -model.loglevel critical" #oper point doesn't matter because HSD is in the common part
    QUANTIZATION_SERVICE_PARAMS="--dumped_data_dir ${LOCATION_OF_PREPARED_DATA} --dumped_data_subdirs bop hop"
    QUANTIZATION_BITDEPTH_PARAMS="--weights_bd $WEIGHT_BD"
    MODELS_PARAMS="-model.CCS_SGMM.tools_common.model_common.common_modules.ckpt_model_name VM_common --skip_models_check"
    CMD="python -m src.quant.scripts.eval ${CODEC_PARAMS} ${QUANTIZATION_SERVICE_PARAMS} ${QUANTIZATION_BITDEPTH_PARAMS} ${MODELS_PARAMS}"
    echo $CMD
    $CMD
    echo " === Model quantization done === "
fi
