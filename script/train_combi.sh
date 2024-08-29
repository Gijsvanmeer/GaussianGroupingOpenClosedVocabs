#!/bin/bash

# Check if the user provided an argument
if [ "$#" -ne 8 ]; then
    echo "Usage: $0 <dataset_name>"
    echo "$1 <scale>"
    echo "$2 <output_name>"
    echo "$3 <closed_vocab_path>"
    echo "$4 <save_file>"
    echo "$5 <lamda>"
    echo "$6 <use_closed_vocab>"
    echo "$7 <method>"
    exit 1
fi


dataset_name="$1"
scale="$2"
output_name="$3"
dataset_folder="data/$dataset_name"
vocab_path="$4"
save_file="$5"
lamda="$6"
dl3="$7"
method="$8"

if [ ! -d "$dataset_folder" ]; then
    echo "Error: Folder '$dataset_folder' does not exist."
    exit 2
fi


# Gaussian Grouping training

CUDA_LAUNCH_BLOCKING=1 python train_combined_losses.py    -s $dataset_folder -r ${scale}  -m output/${output_name} --config_file config/gaussian_dataset/train.json --save_param True --save_file ${save_file} --closed_vocab_path ${vocab_path} --use_dl3 ${dl3} --method ${method} --lmd ${lamda}

