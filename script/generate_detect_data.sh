#!/bin/bash

datasets="code math"
source_models="llama2 llama3"
target_models="bloom mistral"
detects="vanilla LIDet"

for source_model in $source_models; do
    for target_model in $target_models; do
        for dataset in $datasets; do
            python -u ../generate_detection_data.py --wm 0 --target_name $target_model --source_name $source_model --dataset $dataset --detect "vanilla"
        done
    done
done

for source_model in $source_models; do
    for target_model in $target_models; do
        for dataset in $datasets; do
            for i in {0..9}; do
                python -u ../generate_detection_data.py --wm 0 --target_name $target_model --source_name $source_model --dataset $dataset --detect "LIDet" --hash_key_idx $i
            done
        done
    done
done

for detect in $detects; do
    for source_model in $source_models; do
        for target_model in $target_models; do
            for dataset in $datasets; do   
                for i in {0..9}; do
                    python -u ../generate_detection_data.py --wm 1 --hash_key_idx $i --target_name $target_model --source_name $source_model --dataset $dataset --detect $detect --delta 3.0
                done
            done
        done
    done
done
