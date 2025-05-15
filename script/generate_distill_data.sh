#!/bin/bash

datasets="code math"
source_models="llama2 llama3"

for source_model in $source_models; do
    for dataset in $datasets; do
        python -u ../generate_distill_data.py --model_name $source_model --dataset $dataset --wm 0
    done
done

for source_model in $source_models; do
    for dataset in $datasets; do
        for i in {0..9}; do
            python -u ../generate_distill_data.py --model_name $source_model --dataset $dataset --wm 1 --hash_key_idx $i --delta 3.0
        done
    done
done


