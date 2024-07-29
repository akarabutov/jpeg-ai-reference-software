#!/bin/bash

path=$1
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

for f in ${path}/*.pth
do
    echo ${f}
    python ${SCRIPT_DIR}/cleanup_cp.py ${f} ${f} --exclude best_loss optimizer
done