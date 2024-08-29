#!/bin/bash

# Check if the user provided an argument
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dataset_name>"
    echo "Usage: $1 <vocab_path>"
    exit 1
fi


dataset_name="$1"
vocab_path="$2"

# Segmentation rendering using trained model
python render_closed_vocab.py -m output/${dataset_name} --num_classes 256 --closed_vocab_path ${vocab_path}
