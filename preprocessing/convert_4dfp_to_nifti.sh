#!/bin/bash

# Author: Francesco Chiumento
# License: MIT

# 4dfp to NIfTI Converter
#
# Batch converts 4dfp medical imaging files to NIfTI format.
# Recursively processes all .4dfp.img files in a directory tree.
#
# Input: Directory containing .4dfp.img and .ifh files
# Output: Converted .nii files in same locations
#========================================

# Base directory to search for files
BASE_DIR="/path/to/data/PUP_PIB_msum_SUVR"
CONVERTER="/path/to/tools/nifti_4dfp/nifti_4dfp"


# check if converter program exists
if [ ! -f "$CONVERTER" ]; then
    echo "Error: nifti_4dfp converter not found at $CONVERTER"
    exit 1
fi

# find all .4dfp.img files and convert them
find "$BASE_DIR" -type f -name "*.4dfp.img" | while read -r IMG_FILE; do

    # find corresponding IFH file
    IFH_FILE="${IMG_FILE%.img}.ifh"
    
    # create output nifti filename
    NIFTI_FILE="${IMG_FILE%.4dfp.img}.nii"

    # if nifti file exists skip conversion
    if [ -f "$NIFTI_FILE" ]; then
        echo "File $NIFTI_FILE already exists, skipping conversion"
        continue
    fi

    # check if IFH file exists
    if [ ! -f "$IFH_FILE" ]; then
        echo "Warning, file IFH not found $IMG_FILE, skipping this file"
        continue
    fi

    # conversion
    echo "converting $IMG_FILE -> $NIFTI_FILE"
    $CONVERTER -n "$IFH_FILE" "$NIFTI_FILE"

    # check if conversion succeeded
    if [ -f "$NIFTI_FILE" ]; then
        echo "Conversion completed: $NIFTI_FILE"
    else
        echo "Error during conversion of $IMG_FILE"
    fi
done

echo "Process completed"
