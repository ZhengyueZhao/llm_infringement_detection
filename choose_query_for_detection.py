from peft import AutoPeftModelForCausalLM
from peft import PeftModel
from transformers import AutoTokenizer
from transformers import GenerationConfig, LogitsProcessorList
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
import json
from watermark_processor.extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
import argparse, math
import scipy.stats
import random
import numpy as np
import math
from tqdm import tqdm, trange

###############################################

def get_green_ratio(text, green_list, tokenizer):
    tokens_for_calc = tokenizer.tokenize(text)
    ids_for_calc = tokenizer.convert_tokens_to_ids(tokens_for_calc)
    total_token_num = len(ids_for_calc)
    if total_token_num == 0:
        return -1
    green_token_num = 0
    for ids in ids_for_calc:
        if ids in green_list:
            green_token_num += 1
    return green_token_num/total_token_num

def get_entropy(text, tokenizer):
    tokens_for_calc = tokenizer.tokenize(text)
    ids_for_calc = tokenizer.convert_tokens_to_ids(tokens_for_calc)
    total_token_num = len(ids_for_calc)
    if total_token_num == 0:
        return -1
    ids_reg_list = []
    ids_num_list = []
    for ids in ids_for_calc:
        if ids not in ids_reg_list:
            ids_reg_list.append(ids)
            ids_num_list.append(1)
        else:
            ids_num_list[ids_reg_list.index(ids)]+=1
    num = len(ids_num_list)
    ids_freq_list = [ids_num_list[i]/total_token_num for i in range(num)]
    return -sum([i*math.log(i) for i in ids_freq_list])


if __name__ == '__main__':
    random.seed(776)
    hash_key_list = [random.randint(0, 2**31 - 1) for _ in range(10)]

    parser = argparse.ArgumentParser(description='selection')
    parser.add_argument('--model_name', type=str, default="llama2")
    parser.add_argument('--anchor_models', type=list, default=["llama2", "llama3"])
    parser.add_argument('--delta', type=float, default=2.0)
    parser.add_argument('--hash_key_idx', type=int, default=0)
    parser.add_argument('--anchor_data_num', type=int, default=5000)
    parser.add_argument('--dataset', type=str, default="alpaca")

    args = parser.parse_args()

    print("Begin Selection")

    anchor_data = {}
    for anchor_model in args.anchor_models:
        anchor_data_path = "./anchor_data/{}_{}_no_wm.json".format(anchor_model, args.dataset)
        with open(anchor_data_path, "r") as f:
            anchor_data[anchor_model] = json.load(f)

    anchor_data_path_chatgpt = "./data/stanford_alpaca/alpaca.json"
    with open(anchor_data_path, "r") as f:
        anchor_data["chatgpt"] = json.load(f)

    if args.model_name == "llama2":
        tokenizer_path = "meta-llama/Llama-2-7b-chat-hf"
    elif args.model_name == "llama3":
        tokenizer_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer_detect = AutoTokenizer.from_pretrained(tokenizer_path)

    watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer_detect.get_vocab().values()),
                                                gamma=0.25,
                                                delta=args.delta,
                                                new_hash_key=hash_key_list[args.hash_key_idx],
                                                seeding_scheme="static")
    ### 1. Calculate Entropy of Anchor Dataset for each Query
    print("Calculate Entropy of Anchor Dataset for each Query")
    entropy_list = []
    skip_idx = []
    for anchor_data_idx in tqdm(range(args.anchor_data_num), total=args.anchor_data_num):
        entropy_anchor_models = [get_entropy(anchor_data[reg]['output'][anchor_data_idx], tokenizer_detect) for reg in anchor_data]
        if -1 in entropy_anchor_models:
            skip_idx.append(anchor_data_idx)
        entropy_list.append(np.mean(entropy_anchor_models))
 

    ## 2. Calculate Green-Ratio-Difference of Anchor Models for each Query
    print("Calculate Green-Ratio-Difference of Anchor Models for each Query")
    var_list = []
    green_token_ids = watermark_processor.get_greenlist_ids()
    green_token_ids_list = green_token_ids.cpu().numpy().tolist()
    for anchor_data_idx in tqdm(range(args.anchor_data_num), total=args.anchor_data_num):
        green_ratio_anchor_models = [get_green_ratio(anchor_data[reg]['output'][anchor_data_idx], green_token_ids_list, tokenizer_detect) for reg in anchor_data]
        if -1 in green_ratio_anchor_models:
            skip_idx.append(anchor_data_idx)
        var_list.append(np.var(green_ratio_anchor_models))



    ### 3. Search Target Queries for Detection
    var_max = max(var_list)
    var_min = min(var_list)
    entropy_max = max(entropy_list)
    entropy_min = min(entropy_list)
    value_list = [(var_list[i]-var_min)/(var_max-var_min)-2*(entropy_list[i]-entropy_min)/(entropy_max-entropy_min) for i in range(args.anchor_data_num)]

    for idx in range(len(value_list)):
        if idx in skip_idx:
            value_list[idx] = 999999
    target_query_priority = list(np.argsort(np.array(value_list)))
    target_query_priority = [int(reg) for reg in target_query_priority]
    print("End Selection. Saving ...")

    save_path = "./query_priority_for_detection/{}_{}_wm_{}.json".format(args.model_name, args.dataset, args.hash_key_idx)
    save_file = json.dumps(target_query_priority)
    f = open(save_path, 'w')
    f.write(save_file)