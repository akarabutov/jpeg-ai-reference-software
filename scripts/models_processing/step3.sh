#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=${SCRIPT_DIR}/../..

cd $BASE_DIR
rm -R $BASE_DIR/models/VM_common_int
mkdir -p $BASE_DIR/models/VM_common_int
cp $BASE_DIR/models/VM_common/*.pth $BASE_DIR/models/VM_common_int/
$BASE_DIR/scripts/progressive_decoding/reorder.sh
cp $BASE_DIR/models/VM_common_int/*.pth $BASE_DIR/models/VM_common
