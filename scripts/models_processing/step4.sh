#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=${SCRIPT_DIR}/../..

cd $BASE_DIR
rm -R ${BASE_DIR}/models/VM_common_int
CUDA_VISIBLE_DEVICES="0" $BASE_DIR/scripts/quantize_model.sh