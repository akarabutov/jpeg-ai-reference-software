#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=${SCRIPT_DIR}/../..
# Name of training stage that is the source of models
SOURCE_STAGE_NAME="MSE_VariableRate_12"

for beta_path in ${BASE_DIR}/models/VM_common/train_stages/${SOURCE_STAGE_NAME}/*
do
    beta=`basename ${beta_path}`
    echo "Process ${beta}"
    cp ${beta_path}/best.pth ${beta}.pth
    python ${SCRIPT_DIR}/../split_cp.py ${beta}.pth ${BASE_DIR}/models
    rm ${beta}.pth
done
