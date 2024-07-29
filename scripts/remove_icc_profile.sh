#!/bin/bash

FILES=$1

for f in $FILES
do
    echo "Process file ${f}"
    convert  ${f} +profile "*" ${f}
done