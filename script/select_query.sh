#!/bin/bash

source_models="llama2 llama3"

for source_model in $source_models; do
    for i in {0..9}; do
        python -u ../choose_query_for_detection.py --model_name $source_model --hash_key_idx $i
    done
done