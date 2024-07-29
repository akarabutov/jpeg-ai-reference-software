#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

steps=("Remove the current models" \
       "Split models from trained ones" \
       "Reduce number of Z distributions" \
       "Reorder weights for parallel decoding" \
       "Quantize models" \
       "Pack weights for uploading")

for idx in {0..5}
do
    echo "=== Step ${idx}: ${steps[$idx]} == "
    ${SCRIPT_DIR}/step${idx}.sh > step${idx}.log 2>&1 
done