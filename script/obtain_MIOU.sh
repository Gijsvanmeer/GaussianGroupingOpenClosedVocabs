#!/bin/bash

# Check if the user provided an argument
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <output_name>"
    echo "$1 <file_name>"
    echo "$2 <use_dl3>"
    echo "$3 <closed_vocab_path>"
    exit 1
fi


output_name="$1"
filename="$2"
use_dl3="$3"
closed_vocab_path="$4"


# Gaussian Grouping training

python calculate_MIoU.py -m output/${output_name} --num_classes 256 --use_dl3 ${use_dl3}  --file_name ${filename} --closed_vocab_path ${closed_vocab_path}
