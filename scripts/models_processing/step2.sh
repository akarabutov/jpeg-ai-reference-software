#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=${SCRIPT_DIR}/../..

cd $BASE_DIR
python $BASE_DIR/scripts/reduce_z_distributions.py