#!/bin/bash
# The copyright in this software is being made available under the BSD
#  License, included below. This software may be subject to other third party
#  and contributor rights, including patent rights, and no such rights are
#  granted under this license.
#
#  Copyright (c) 2010-2026, ITU/ISO/IEC
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
#  be used to endorse or promote products derived from this software without
#  specific prior written permission.
#
# DISCLAIMER
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
#  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
#  THE POSSIBILITY OF SUCH DAMAGE.
#
# -----------------------------------------------------------------------------
# Downloads the JPEG AI training/validation dataset.
#
# The dataset used to live on an sFTP server (amalia.img.lx.it.pt) behind a
# password. That server is gone; the files were moved to two public mirrors:
#
#   ITU : https://www.itu.int/wftp3/Public/t/testsignal/SpeImage/T840-3/v2026_01/
#   ISO : https://standards.iso.org/iso-iec/6048/-3/ed-1/en/
#
# Both mirrors serve the same files over plain HTTPS (no password needed).
# This script lets you pick which mirror to use and which dataset categories
# to download.
# -----------------------------------------------------------------------------

set -eu
set -o pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATA_DIR="${SCRIPT_DIR}/../data"

ITU_BASE="https://www.itu.int/wftp3/Public/t/testsignal/SpeImage/T840-3/v2026_01"
ISO_BASE="https://standards.iso.org/iso-iec/6048/-3/ed-1/en"

# -----------------------------------------------------------------------------
# Category catalogue
# -----------------------------------------------------------------------------
ALL_CATEGORIES=(natural_cropped natural_full scc extra valid_cropped valid_full)

declare -A CAT_LABEL=(
    [natural_cropped]="Natural images, training set - CROPPED (ready-to-use 512x512 random crops)"
    [natural_full]="Natural images, training set - FULL resolution (non-cropped source images, merged into the same training folder - run src/codec/datasets/image_crop.py yourself if you need uniform crops)"
    [scc]="Screen-content training patches (SCC7000), merged into the training folder"
    [extra]="Extra training categories (CP50, EXCEL300, HF2000, LQ7000, MD300, PHF200, PHFA500), merged into the training folder"
    [valid_cropped]="Validation set - CROPPED"
    [valid_full]="Validation set - FULL resolution, merged into the same validation folder"
)

declare -A CAT_SIZE=(
    [natural_cropped]="175.7 GB (34 archive parts, 4 batches)"
    [natural_full]="57.7 GB (11 archive parts)"
    [scc]="1.4 GB (1 file)"
    [extra]="4.4 GB (7 files)"
    [valid_cropped]="4.0 GB (1 file)"
    [valid_full]="4.0 GB (1 file)"
)

# Prints the list of remote file names that make up a category, one per line.
files_for_category() {
    case "$1" in
        natural_cropped)
            for batch_parts in "00000-01299:9" "01300-02599:9" "02600-03899:9" "03900-05263:7"; do
                batch="${batch_parts%%:*}"
                parts="${batch_parts##*:}"
                for ((p=1; p<=parts; p++)); do
                    printf "jpegai_train-natural-cropped_%s_bundle.7z.%03d\n" "$batch" "$p"
                done
            done
            ;;
        natural_full)
            for ((p=1; p<=11; p++)); do
                printf "jpegai_train-natural-full_bundle.7z.%03d\n" "$p"
            done
            ;;
        scc)
            echo "jpegai_train-extra_SCC7000P2.tar"
            ;;
        extra)
            for name in CP50 EXCEL300 HF2000 LQ7000 MD300 PHF200 PHFA500; do
                echo "jpegai_train-extra_${name}.tar"
            done
            ;;
        valid_cropped)
            echo "jpegai_valid-natural_cropped_bundle.7z.001"
            ;;
        valid_full)
            echo "jpegai_valid-natural-full_bundle.7z.001"
            ;;
        *)
            echo "Unknown category: $1" >&2
            exit 1
            ;;
    esac
}

# -----------------------------------------------------------------------------
# Usage / argument parsing
# -----------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: $(basename "$0") [--mirror itu|iso] [--sets cat1,cat2,...] [--downloader curl|wget] [--debug] [--list] [-y|--yes] [-h|--help]

  --mirror itu|iso        Which server to download from (skips the interactive prompt).
  --sets LIST             Comma-separated category ids to download (skips the interactive prompt).
  --downloader curl|wget  Force a specific download tool. Default: curl if available, else wget.
  --debug                 Don't download or extract anything; just print the command that would
                          be run for every file.
  --list                  Print the available categories and sizes, then exit.
  -y, --yes               Don't ask for confirmation before downloading.
  -h, --help              Show this help.

Available categories:
EOF
    for cat in "${ALL_CATEGORIES[@]}"; do
        printf "  %-14s %-10s %s\n" "$cat" "${CAT_SIZE[$cat]}" "${CAT_LABEL[$cat]}"
    done
}

MIRROR=""
SETS=""
DOWNLOADER=""
ASSUME_YES=0
DEBUG=0

while [ $# -gt 0 ]; do
    case "$1" in
        --mirror) MIRROR="$2"; shift 2 ;;
        --sets) SETS="$2"; shift 2 ;;
        --downloader) DOWNLOADER="$2"; shift 2 ;;
        --debug) DEBUG=1; shift ;;
        --list) usage; exit 0 ;;
        -y|--yes) ASSUME_YES=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

# -----------------------------------------------------------------------------
# Downloader selection: curl by default, fall back to wget if curl is missing.
# -----------------------------------------------------------------------------
if [ -n "$DOWNLOADER" ]; then
    case "$DOWNLOADER" in
        curl|wget) ;;
        *) echo "Unknown downloader '$DOWNLOADER' (expected curl or wget)." >&2; exit 1 ;;
    esac
    if ! command -v "$DOWNLOADER" &> /dev/null; then
        echo "Requested downloader '$DOWNLOADER' is not installed." >&2
        exit 1
    fi
elif command -v curl &> /dev/null; then
    DOWNLOADER="curl"
elif command -v wget &> /dev/null; then
    DOWNLOADER="wget"
else
    echo "Neither 'curl' nor 'wget' is available. Install one of them and re-run this script." >&2
    exit 1
fi
echo "Using downloader: $DOWNLOADER"

# -----------------------------------------------------------------------------
# Interactive mirror selection
# -----------------------------------------------------------------------------
if [ -z "$MIRROR" ]; then
    echo "Where do you want to download the dataset from?"
    echo "  1) ITU  - ${ITU_BASE}"
    echo "  2) ISO  - ${ISO_BASE}"
    read -rp "Enter 1 or 2: " choice
    case "$choice" in
        1) MIRROR="itu" ;;
        2) MIRROR="iso" ;;
        *) echo "Invalid choice." >&2; exit 1 ;;
    esac
fi

case "$MIRROR" in
    itu) BASE_URL="$ITU_BASE" ;;
    iso) BASE_URL="$ISO_BASE" ;;
    *) echo "Unknown mirror '$MIRROR' (expected itu or iso)." >&2; exit 1 ;;
esac
echo "Using mirror: $BASE_URL"
echo

# -----------------------------------------------------------------------------
# Interactive category selection
# -----------------------------------------------------------------------------
SELECTED=()
if [ -n "$SETS" ]; then
    IFS=',' read -ra SELECTED <<< "$SETS"
else
    echo "Which dataset categories do you want to download?"
    i=1
    declare -A INDEX_TO_CAT
    for cat in "${ALL_CATEGORIES[@]}"; do
        printf "  %d) %-10s - %s\n" "$i" "${CAT_SIZE[$cat]}" "${CAT_LABEL[$cat]}"
        INDEX_TO_CAT[$i]="$cat"
        i=$((i+1))
    done
    echo "  a) all of the above"
    read -rp "Enter numbers separated by spaces (e.g. \"1 3 5\"), or 'a' for all: " answer
    if [ "$answer" = "a" ]; then
        SELECTED=("${ALL_CATEGORIES[@]}")
    else
        for n in $answer; do
            cat="${INDEX_TO_CAT[$n]:-}"
            if [ -z "$cat" ]; then
                echo "Invalid selection: $n" >&2
                exit 1
            fi
            SELECTED+=("$cat")
        done
    fi
fi

if [ ${#SELECTED[@]} -eq 0 ]; then
    echo "Nothing selected, exiting." >&2
    exit 1
fi

echo
echo "Selected categories:"
for cat in "${SELECTED[@]}"; do
    printf "  - %-14s %-10s %s\n" "$cat" "${CAT_SIZE[$cat]}" "${CAT_LABEL[$cat]}"
done
echo

if [ "$DEBUG" -eq 1 ]; then
    echo "--debug: nothing will be downloaded or extracted, printing the download commands only."
elif [ "$ASSUME_YES" -ne 1 ]; then
    read -rp "Proceed with download into ${DATA_DIR}? [y/N] " confirm
    case "$confirm" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; exit 0 ;;
    esac
fi

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# -----------------------------------------------------------------------------
# Download + extraction helpers
# -----------------------------------------------------------------------------
download_file() {
    local fname="$1"
    local url="${BASE_URL}/${fname}"
    local cmd
    case "$DOWNLOADER" in
        curl) cmd=(curl -fL --retry 3 --retry-delay 5 -C - -o "${fname}" "${url}") ;;
        wget) cmd=(wget -c --tries=3 --waitretry=5 -O "${fname}" "${url}") ;;
    esac
    if [ "$DEBUG" -eq 1 ]; then
        printf '%q ' "${cmd[@]}"; echo
    else
        echo "==> Downloading ${fname}"
        "${cmd[@]}"
    fi
}

require_7z() {
    if ! command -v 7z &> /dev/null; then
        echo "The '7z' command is required to extract the downloaded archives but was not found." >&2
        echo "Install p7zip-full (Linux) / p7zip (macOS) / 7-Zip (Windows) and re-run this script." >&2
        exit 1
    fi
}

# Extract a multi-volume 7z archive flat (no subfolders) into $2, like 'unzip -j'.
extract_7z_flat() {
    local first_part="$1" outdir="$2"
    mkdir -p "$outdir"
    7z e -y "$first_part" -o"$outdir" > /dev/null
}

# Writes a flat list of the files in $1 (one per line, skipping any existing
# .txt files) to $2.
generate_file_list() {
    local dir="$1" list_file="$2"
    [ -d "$dir" ] || return 0
    echo "==> Generating ${list_file}"
    ls -1 "$dir" | { grep -v '\.txt$' || true; } > "${list_file}"
}

download_category() {
    local cat="$1"
    while IFS= read -r fname; do
        download_file "$fname"
    done < <(files_for_category "$cat")
}

# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------
for cat in "${SELECTED[@]}"; do
    download_category "$cat"
done

if [ "$DEBUG" -eq 1 ]; then
    echo
    echo "--debug: skipping extraction and file-list generation."
    exit 0
fi

# -----------------------------------------------------------------------------
# Extraction
# -----------------------------------------------------------------------------
NEEDS_7Z=0
for cat in "${SELECTED[@]}"; do
    case "$cat" in
        natural_cropped|natural_full|valid_cropped|valid_full) NEEDS_7Z=1 ;;
    esac
done
[ "$NEEDS_7Z" -eq 1 ] && require_7z

# All training categories land flat in one directory, and all validation
# categories land flat in one other directory - scripts/train.sh and
# cfg/train.json consume a single --data_dir/--lst and --val_data_dir/--val_lst
# pair each, not one per category.
TRAINING_DIR=jpegai_training_set
VALIDATION_DIR=jpegai_validation_set

for cat in "${SELECTED[@]}"; do
    case "$cat" in
        natural_cropped)
            echo "==> Extracting natural_cropped into ${TRAINING_DIR}/"
            for batch in "00000-01299" "01300-02599" "02600-03899" "03900-05263"; do
                extract_7z_flat "jpegai_train-natural-cropped_${batch}_bundle.7z.001" "$TRAINING_DIR"
            done
            ;;
        natural_full)
            echo "==> Extracting natural_full into ${TRAINING_DIR}/"
            echo "    Note: these are full-resolution source images, not pre-cropped."
            echo "    Use src/codec/datasets/image_crop.py (see src/codec/datasets/crop_image.sh"
            echo "    for example invocations) to produce training crops from them."
            extract_7z_flat jpegai_train-natural-full_bundle.7z.001 "$TRAINING_DIR"
            ;;
        scc)
            echo "==> Extracting scc into ${TRAINING_DIR}/"
            mkdir -p "$TRAINING_DIR"
            tar -xf jpegai_train-extra_SCC7000P2.tar -C "$TRAINING_DIR"
            ;;
        extra)
            echo "==> Extracting extra categories into ${TRAINING_DIR}/"
            mkdir -p "$TRAINING_DIR"
            for name in CP50 EXCEL300 HF2000 LQ7000 MD300 PHF200 PHFA500; do
                tar -xf "jpegai_train-extra_${name}.tar" -C "$TRAINING_DIR"
            done
            ;;
        valid_cropped)
            echo "==> Extracting valid_cropped into ${VALIDATION_DIR}/"
            extract_7z_flat jpegai_valid-natural_cropped_bundle.7z.001 "$VALIDATION_DIR"
            ;;
        valid_full)
            echo "==> Extracting valid_full into ${VALIDATION_DIR}/"
            extract_7z_flat jpegai_valid-natural-full_bundle.7z.001 "$VALIDATION_DIR"
            ;;
    esac
done

# -----------------------------------------------------------------------------
# One file list per directory, named the way scripts/train.sh and
# docs/architecture/13-training.md expect them.
# -----------------------------------------------------------------------------
generate_file_list "$TRAINING_DIR" "${TRAINING_DIR}/jpegai_training_set512_random_crop_16.txt"
generate_file_list "$VALIDATION_DIR" "${VALIDATION_DIR}/jpegai_validation_set_10.txt"

echo
echo "Done."
