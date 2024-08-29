#!/bin/bash

# Check if the user provided an argument
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <dataset_name>"
    echo "$1 <scale>"
    echo "$2 <output_name>"
    echo "$3 <save_file>"
    exit 1
fi


dataset_name="$1"
scale="$2"
output_name="$3"
dataset_folder="data/$dataset_name"
save_file="$4"

if [ ! -d "$dataset_folder" ]; then
    echo "Error: Folder '$dataset_folder' does not exist."
    exit 2
fi


# Gaussian Grouping training
CUDA_LAUNCH_BLOCKING=1 python train_default.py    -s $dataset_folder -r ${scale}  -m output/${output_name} --config_file config/gaussian_dataset/train.json --save_param True --save_file ${save_file}
