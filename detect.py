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
from tqdm import tqdm, trange
import os
import re




def calc_token_freq(text_for_calc):
    tokens_for_calc = tokenizer.tokenize(text_for_calc)
    ids_for_calc = tokenizer.convert_tokens_to_ids(tokens_for_calc)
    ids_for_calc_list = ids_for_calc
    total_token_num = len(ids_for_calc_list)
    vocab_num = len(list(tokenizer.get_vocab().values()))
    ids_freq_dict = {}
    for ids in range(vocab_num):
        ids_freq_dict[str(ids)] = 0

    for ids in ids_for_calc_list:
        ids_freq_dict[str(ids)] += 1

    for ids in ids_freq_dict:
        ids_freq_dict[str(ids)] = ids_freq_dict[str(ids)]/total_token_num
    return ids_freq_dict


def get_subset(freq0, freq_w, num=100):
    sub_2_0 = {}
    sub_2_0_abs = {}
    sub_2_1 = {}
    for idx in range(32000):
        if freq0[str(idx)] > 0 and freq_w[str(idx)] > 0:
            sub_2_0[idx] = (freq_w[str(idx)] - freq0[str(idx)])
            sub_2_0_abs[idx] = abs(sub_2_0[idx])
        else:
            sub_2_0[idx] = 0.0
            sub_2_0_abs[idx] = 0.0
    sub_2_0_top = sorted(sub_2_0_abs.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)[:num]
    sub_2_0_top_list = [int(reg[0]) for reg in sub_2_0_top]
    subset = []
    for reg in sub_2_0_top_list:
        subset.append(reg)
    return subset


def stastic_data_with_freq(data, top_list):
    green_ratio = 0
    total_ratio = 0
    res = 0
    for ids in top_list:
        total_ratio += data[str(ids)]
        if ids in green_token_ids_list:
            green_ratio += data[str(ids)]
    res = green_ratio / total_ratio
    return res

def get_gamma_star(green_list, subset):
    green_num = 0
    for ids in subset:
        if ids in green_list:
            green_num += 1
    return green_num/len(subset)

def get_modify_SG_T(text_for_calc, subset, green_list):
    tokens_for_calc = tokenizer.tokenize(text_for_calc)
    ids_for_calc = tokenizer.convert_tokens_to_ids(tokens_for_calc)
    SG = 0
    T = 0
    for ids in ids_for_calc:
        if ids in subset:
            T += 1
            if ids in green_list:
                SG += 1
    return SG, T



def get_z_score(total_words_num, target_words_num, gamma_words):
    return (target_words_num-gamma_words*total_words_num)/math.sqrt(gamma_words*(1-gamma_words)*total_words_num)

def get_z_score_ratio(green_ratio, gamma_words):
    return (green_ratio-gamma_words)/math.sqrt(gamma_words*(1-gamma_words))*math.sqrt(args.length)


def load_tokenizer(model_name):
    if model_name == "llama2":
        tokenizer_path = "meta-llama/Llama-2-7b-chat-hf"
    elif model_name == "llama3":
        tokenizer_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif model_name == "mistral":
        tokenizer_path = "mistralai/Mistral-7B-Instruct-v0.2"
    return tokenizer_path

if __name__ == '__main__':
    random.seed(776)
    hash_key_list = [random.randint(0, 2**31 - 1) for _ in range(10)]

    parser = argparse.ArgumentParser(description='tuning')
    parser.add_argument('--detect', type=str, default="vanilla")
    parser.add_argument('--delta', type=float, default=2.0)
    parser.add_argument('--length', type=int, default=20000)
    parser.add_argument('--source_name_list', nargs='+')
    parser.add_argument('--target_name_list', nargs='+')
    parser.add_argument('--hash_key_idx_list', nargs='+', type=int)
    parser.add_argument('--dataset_list', nargs='+')
    parser.add_argument('--anchor_list', nargs='+')
    args = parser.parse_args()

    print(args.length)
    print(args.source_name_list)
    print(args.target_name_list)
    print(args.hash_key_idx_list)
    print(args.dataset_list)
    print(args.detect)
    print(args.anchor_list)


    anchor_data_path_list = []
    if "llama2" in args.anchor_list:
        anchor_data_path_list.append("./data_for_distill/llama2_alpaca_no_wm.json")
    if "llama3" in args.anchor_list:
        anchor_data_path_list.append("./data_for_distill/llama3_alpaca_no_wm.json")
    if "bloom" in args.anchor_list:
        anchor_data_path_list.append("./data/alpaca_bloom-7b_output_no_wm_0.json")
    if "mistral" in args.anchor_list:
        anchor_data_path_list.append("./data/alpaca_mistral-7b_output_no_wm.json")

    anchor_data_list = []
    for anchor_path in anchor_data_path_list:
        with open(anchor_path, "r") as f:
            anchor_data_tmp = json.load(f)
        anchor_data_list.append(anchor_data_tmp)

    labels = []
    preds = []
    hits = []
    hit_pairs = []

    total_num = len(args.source_name_list)*len(args.target_name_list)*len(args.hash_key_idx_list)*len(args.dataset_list)*2

    num_query = []

    with trange(total_num, desc='detect') as tbar:
        for source_model in args.source_name_list:
            tokenizer = AutoTokenizer.from_pretrained(load_tokenizer(source_model))
            for hash_key_idx in args.hash_key_idx_list:
                    watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                    gamma=0.25,
                                                    delta=args.delta,
                                                    new_hash_key=hash_key_list[hash_key_idx],
                                                    seeding_scheme="static")
                    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                            gamma=0.25, # should match original setting
                                            new_hash_key=hash_key_list[hash_key_idx],
                                            seeding_scheme="static", # should match original setting
                                            device="cuda", # must match the original rng device type
                                            tokenizer=tokenizer,
                                            z_threshold=4.0,
                                            normalizers=[],
                                            ignore_repeated_ngrams=False)

                    green_token_ids = watermark_processor.get_greenlist_ids()
                    green_token_ids_list = green_token_ids.cpu().numpy().tolist()
                    for target_model in args.target_name_list:
                        for dataset in args.dataset_list:
                            pred_pair = []
                            for wm in [0,1]:
                                if args.detect == "vanilla":
                                    if wm == 1:
                                        test_data_path = "./data_for_detection/{}_from_{}_{}_wm_{}_delta_{}_{}.json".format(target_model, source_model, dataset, hash_key_idx, args.delta, "vanilla")
                                    elif wm == 0:
                                        if dataset in ["code", "math"]:
                                            test_data_path = "./data_for_detection/{}_from_{}_{}_no_wm_{}.json".format(target_model, source_model, dataset, "vanilla")
                                            if not os.path.exists(test_data_path):
                                                test_data_path = "./data_for_detection/{}_from_{}_{}_no_wm_{}.json".format("mistral", "llama2", dataset, "vanilla")
                                        else:
                                            test_data_path = "./data_for_detection/{}_from_{}_{}_no_wm_{}.json".format(target_model, source_model, "code", "vanilla")
                                            if not os.path.exists(test_data_path):
                                                test_data_path = "./data_for_detection/{}_from_{}_{}_no_wm_{}.json".format("mistral", "llama2", "code", "vanilla")
                                    with open(test_data_path, "r") as f:
                                        test_data = json.load(f)

                                    text_for_detect = ''
                                    flag = 0
                                    for text in test_data['output']:
                                        text_for_detect += text
                                        tokens_for_calc = tokenizer.tokenize(text_for_detect)
                                        ids_for_calc = tokenizer.convert_tokens_to_ids(tokens_for_calc)
                                        total_token_num = len(ids_for_calc)
                                        flag += 1
                                        if total_token_num >= args.length:
                                            num_query.append(flag)
                                            break
                                    # tokens_for_calc = tokenizer.tokenize(text_for_detect)
                                    # ids_for_calc = tokenizer.convert_tokens_to_ids(tokens_for_calc)
                                    # visited_ids_list = []
                                    # total_token_num = 0
                                    # green_token_num = 0
                                    # for ids in ids_for_calc:
                                    #     if ids not in visited_ids_list:
                                    #         total_token_num += 1
                                    #         visited_ids_list.append(ids)
                                    #         if ids in green_token_ids_list:
                                    #             green_token_num += 1
                                        
                                    # z_score = get_z_score(total_token_num, green_token_num, 0.25)
                                    green_ratio = watermark_detector.detect(text_for_detect)['green_fraction']
                                    z_score = get_z_score_ratio(green_ratio, 0.25)
                                    # print("wm {}: z {}".format(wm, z_score))

                                    # print("z-score: ", z_score)
                                    if z_score>=4.0:
                                        pred = 1
                                    else:
                                        pred = 0
                                    

                                elif args.detect == "LIDet":
                                    if wm == 1:
                                        test_data_path = "./data_for_detection/{}_from_{}_{}_wm_{}_delta_{}_{}.json".format(target_model, source_model, dataset, hash_key_idx, args.delta, args.detect)
                                        if not os.path.exists(test_data_path):
                                            test_data_path = "./data_for_detection/{}_from_{}_{}_wm_{}_delta_{}_{}.json".format(target_model, source_model, dataset, hash_key_idx, args.delta, "LIDet")
                                            
                                    elif wm == 0:
                                        if dataset in ["code", "math"]:
                                            test_data_path = "./data_for_detection/{}_from_{}_{}_no_wm_{}_{}.json".format(target_model, source_model, dataset, hash_key_idx, args.detect)
                                            if not os.path.exists(test_data_path):
                                                test_data_path = "./data_for_detection/{}_from_{}_{}_no_wm_{}_{}.json".format(target_model, source_model, dataset, hash_key_idx, "LIDet")
                                        else:
                                            test_data_path = "./data_for_detection/{}_from_{}_{}_no_wm_{}_{}.json".format(target_model, source_model, "code", hash_key_idx, args.detect)
                                            if not os.path.exists(test_data_path):
                                                test_data_path = "./data_for_detection/{}_from_{}_{}_no_wm_{}_{}.json".format(target_model, source_model, "code", hash_key_idx, "LIDet")
                                    with open(test_data_path, "r") as f:
                                        test_data = json.load(f)
                                    text_for_detect = ''
                                    flag = 0
                                    for text in test_data['output']:
                                        text_for_detect += text
                                        tokens_for_calc = tokenizer.tokenize(text_for_detect)
                                        ids_for_calc = tokenizer.convert_tokens_to_ids(tokens_for_calc)
                                        total_token_num = len(ids_for_calc)
                                        flag += 1
                                        if total_token_num >= args.length:
                                            num_query.append(flag)
                                            break
                                    tokens_for_calc = tokenizer.tokenize(text_for_detect)
                                    ids_for_calc = tokenizer.convert_tokens_to_ids(tokens_for_calc)
                                    total_token_num = len(ids_for_calc)
                                    green_token_num = 0
                                    for ids in ids_for_calc:
                                        if ids in green_token_ids_list:
                                            green_token_num += 1
                                    z_score = get_z_score(total_token_num, green_token_num, 0.25)
                                    anchor_data = ''
                                    query_priority_list_path = "./query_priority_for_detection/{}_{}_wm_{}.json".format(source_model, "alpaca", hash_key_idx)
                                    with open(query_priority_list_path, "r") as f:
                                        query_priority_list = json.load(f)
                                    for idx in query_priority_list[:100]:
                                            for anchor_idx in range(len(anchor_data_list)):
                                                anchor_data += anchor_data_list[anchor_idx]['output'][idx]
                                    tokens_anchor = tokenizer.tokenize(anchor_data)
                                    ids_anchor = tokenizer.convert_tokens_to_ids(tokens_anchor)
                                    total_token_num_anchor = len(ids_anchor)
                                    green_token_num_anchor = 0
                                    for ids in ids_anchor:
                                        if ids in green_token_ids_list:
                                            green_token_num_anchor += 1
                                    z_score_anchor = get_z_score_ratio(green_token_num_anchor/total_token_num_anchor, 0.25)
                                    if z_score>=z_score_anchor+4.0:
                                        pred = 1
                                    else:
                                        pred = 0

                                pred_pair.append(pred)
                                preds.append(pred)
                                hits.append(pred==wm)
                                labels.append(wm)
                                acc = hits.count(True)/len(hits)
                                if len(hit_pairs)>0:
                                    dsr = hit_pairs.count(True)/len(hit_pairs)
                                else:
                                    dsr = 0
                                tbar.set_postfix({'acc':acc, 'dsr':dsr})
                                tbar.update()
                            if pred_pair[0]==0 and pred_pair[1]==1:
                                hit_pairs.append(True)
                            else:
                                hit_pairs.append(False)

    ACC = hits.count(True)/len(hits)
    DSR = hit_pairs.count(True)/len(hit_pairs)

    TPR = 0
    for idx in range(len(labels)):
        if labels[idx]==1 and hits[idx]==True:
            TPR += 1
    TPR = TPR/(len(labels)/2)
    TNR = 0
    for idx in range(len(labels)):
        if labels[idx]==0 and hits[idx]==True:
            TNR += 1
    TNR = TNR/(len(labels)/2)

    print("# Sample: ", len(labels))
    print("ACC: ", ACC)
    print("TPR: ", TPR)
    print("TNR: ", TNR)
    print("DSR: ", DSR)
    print("AVG #Q: ", sum(num_query)/len(num_query))





