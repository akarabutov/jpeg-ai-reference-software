#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=${SCRIPT_DIR}/../..

# Remove the current files
for d in VM_common VM_common_int VM_bop VM_hop VM_sop
do
    git rm -r -f ${BASE_DIR}/models/${d}/*.dvc
done
