#!/bin/bash

python -u ../generate_anchor_data.py --model_name "llama2" --dataset "alpaca"
python -u ../generate_anchor_data.py --model_name "llama3" --dataset "alpaca"
