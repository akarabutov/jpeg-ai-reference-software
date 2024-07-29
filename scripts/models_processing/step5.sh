#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=${SCRIPT_DIR}/../..

ADDITIONAL_ARGS="--pack"

cd $BASE_DIR

# Process models
for d in VM_bop VM_hop VM_sop
do
    python ${BASE_DIR}/scripts/process_model.py ${d} ${ADDITIONAL_ARGS}
done

# Process models
for d in VM_common VM_common_int
do
    python ${BASE_DIR}/scripts/process_model.py ${d} ${ADDITIONAL_ARGS} --exclude-keys ""
done