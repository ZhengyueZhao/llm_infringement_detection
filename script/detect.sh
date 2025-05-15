#!/bin/bash


python -u ../detect.py \
    --delta 3.0 \
    --source_name_list "llama2" "llama3"\
    --target_name_list "bloom" "mistral"\
    --hash_key_idx_list 0 1 2 3 4 5 6 7 8 9 \
    --dataset_list "code" "math"\
    --detect "vanilla" \
    --anchor_list "llama2" "llama3"

python -u ../detect.py \
    --delta 3.0 \
    --source_name_list "llama2" "llama3"\
    --target_name_list "bloom" "mistral"\
    --hash_key_idx_list 0 1 2 3 4 5 6 7 8 9 \
    --dataset_list "code" "math"\
    --detect "LIDet" \
    --anchor_list "llama2" "llama3"
