#!/bin/bash

N=10
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ${SCRIPT_DIR}/..
mkdir -p data/test_${N}bit
for s in data/test/*.png
do
    a=`basename $s`
    Nbits="${N}bit"
    b=$(echo $a | sed "s/8bit/${Nbits}/g")
    of="data/test_${Nbits}/${b}"
    #convert test/$a -colorspace RGB -depth ${N} PNG48:test_${N}bit/$b
    echo "Process $s -> $of"
    rs=`realpath $s`
    rof=`realpath $of`
    python -m scripts.convert_bitdepth $rs $rof ${N} --type 0
done