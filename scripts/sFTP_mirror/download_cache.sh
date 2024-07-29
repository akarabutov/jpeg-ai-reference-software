#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

OUTPUT_BASE_DIR=${SCRIPT_DIR}/../../.dvc/
if (( $# > 0 )); then
    OUTPUT_BASE_DIR=$1
fi

wget --mirror -pc --convert-links --reject "index.html*" -nH --no-parent -e robots=off -k -P  ${OUTPUT_BASE_DIR} 'https://jpeg-git.lx.it.pt/cache/'
