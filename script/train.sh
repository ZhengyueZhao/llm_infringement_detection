#!/bin/bash

datasets="code math"
source_models="llama2 llama3"
target_models="bloom mistral"

for source_model in $source_models; do
    for target_model in $target_models; do
        for dataset in $datasets; do
            python -u ../train.py --target_name $target_model --source_name $source_model --dataset $dataset --wm 0
        done
    done
done

for source_model in $source_models; do
    for target_model in $target_models; do
        for dataset in $datasets; do
            for i in {0..9}; do
                python -u ../train.py --target_name $target_model --source_name $source_model  --dataset $dataset --wm 1 --hash_key_idx $i --delta 3.0
            done
        done
    done
done