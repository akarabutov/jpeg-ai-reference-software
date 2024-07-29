#!/bin/bash

usage() { echo "Usage: $0 [-p <hop|bop>] [-l <mse|mss>] [-q <012|025|050|075|100>]" 1>&2; exit 1; }

while getopts ":p:l:q:" o; do
    case "${o}" in
        l)
            l=${OPTARG}
            ((l == "mse" || l == "mss")) || usage
            ;;
        q)
            q=${OPTARG}
            ((q == "012" || q == "025" || q == "050" || q == "075" || q == "100")) || usage
            ;;
        p)
            p=${OPTARG}
            ((p == "hop" || p == "bop")) || usage
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${l}" ] || [ -z "${q}" ] || [ -z "${p}" ]; then
    usage
fi

echo "OP = ${p}"
echo "LOSS = ${l}"
echo "QP = ${q}"
echo "python train.py -opt options/train/eicci_${p}_${l}_${q}.yml"

python train.py -opt options/train/eicci_${p}_${l}_${q}.yml
