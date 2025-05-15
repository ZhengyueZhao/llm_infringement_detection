import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModel, MistralForCausalLM
from transformers import GenerationConfig, LogitsProcessorList
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import AutoPeftModelForCausalLM
from peft import PeftModel
import torch
import torch.nn.functional as F
import gc
import numpy as np
import torch.nn as nn
import time
import argparse
import os
from tqdm import tqdm, trange
import json
import random
import sys
from watermark_processor.extended_watermark_processor import WatermarkLogitsProcessor

TEMPLATE = (
    "[INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n"
    "{instruction} [/INST]"
)
DEFAULT_SYSTEM_PROMPT = """ you are a helpful assistant. """



def generate_prompt(instruction, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return TEMPLATE.format_map({'instruction': instruction,'system_prompt': system_prompt})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tuning')
    parser.add_argument('--wm', type=float, default=1, help="watermart ratio")
    parser.add_argument('--detect', type=str, default="vanilla")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--delta', type=float, default=3.0)
    parser.add_argument('--source_name', type=str, default="llama2", help="model name")
    parser.add_argument('--target_name', type=str, default="bloom", help="model name")
    parser.add_argument('--hash_key_idx', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="code", help="dataset")
    args = parser.parse_args()
    wm = args.wm

    ## Target Model
    if args.target_name == "mistral":
        model_path = "mistralai/Mistral-7B-Instruct-v0.2"
    elif args.target_name == "bloom":
        model_path = "bigscience/bloom-7b1"

    ## Lora Model
    if wm == 0:
        peft_path = "./distilled_models/{}_from_{}_{}_no_wm.json".format(args.target_name, args.source_name, args.dataset)
    else:
        peft_path = "./distilled_models/{}_from_{}_{}_wm_{}_delta_{}.json".format(args.target_name, args.source_name, args.dataset, args.hash_key_idx, args.delta)

    device = 'cuda'
    data_path = "./data/stanford_alpaca/alpaca_data.json"

    ## Save Path
    if wm == 0:
        if args.detect == "vanilla":
            OUTPUT_DIR = "./data_for_detection/{}_from_{}_{}_no_wm_{}.json".format(args.target_name, args.source_name, args.dataset, args.detect)
        elif args.detect=="LIDet":
            OUTPUT_DIR = "./data_for_detection/{}_from_{}_{}_no_wm_{}_{}.json".format(args.target_name, args.source_name, args.dataset, args.hash_key_idx, args.detect)
    else:
        if args.detect == "vanilla":
            OUTPUT_DIR = "./data_for_detection/{}_from_{}_{}_wm_{}_delta_{}_{}.json".format(args.target_name, args.source_name, args.dataset, args.hash_key_idx, args.delta, args.detect)
        elif args.detect=="LIDet":
            OUTPUT_DIR = "./data_for_detection/{}_from_{}_{}_wm_{}_delta_{}_{}.json".format(args.target_name, args.source_name, args.dataset, args.hash_key_idx, args.delta, args.detect)

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
 
    
    with open(data_path) as f:
        alpaca_data = json.load(f)
    
    query_list = [alpaca_data_reg['instruction']+' '+alpaca_data_reg['input'] for alpaca_data_reg in alpaca_data]


    # initialize model
    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                    ).to(device).eval()
    model = PeftModel.from_pretrained(model, peft_path)

    tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        use_fast=False
                    )
    
    generation_config = GenerationConfig(
                                    temperature=0.8,
                                    top_k=40,
                                    top_p=0.95,
                                    do_sample=True,
                                    num_beams=1,
                                    repetition_penalty=1.1,
                                    max_new_tokens=512
                                )
    

    output_list = {'input':[], 'output':[]}
    if args.detect == "vanilla":
        random.seed(args.seed)
        query_idx_list = [random.randint(0, 5000) for _ in range(1000)]
    elif args.detect == "LIDet":
        query_priority_path = "./query_priority_for_detection/{}_{}_wm_{}.json".format(args.source_name, "alpaca", args.hash_key_idx)
        with open(query_priority_path) as f:
            query_idx_list = json.load(f)
            query_idx_list = query_idx_list[:1000]
    concact_response = ''
    print("#"*20)

    for query_idx in query_idx_list:
        input_text = generate_prompt(instruction=query_list[query_idx], system_prompt=DEFAULT_SYSTEM_PROMPT)
        inputs = tokenizer(input_text, return_tensors="pt")

        generation_output = model.generate(
            input_ids = inputs["input_ids"].to(device),
            attention_mask = inputs['attention_mask'].to(device),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            generation_config = generation_config
        )

        s = generation_output[0]
        output = tokenizer.decode(s,skip_special_tokens=True)
        response = output.split("[/INST]")[-1].strip()
        # print(response)
        output_list['input'].append(query_list[query_idx])
        output_list['output'].append(response)


        concact_response = concact_response + response
        tokens_for_calc = tokenizer.tokenize(concact_response)
        ids_for_calc = tokenizer.convert_tokens_to_ids(tokens_for_calc)
        total_token_num = len(ids_for_calc)
        print("# Token: ", total_token_num)
        if total_token_num >= 20000:
            break



    save_file = json.dumps(output_list)
    f = open(OUTPUT_DIR, 'w')
    f.write(save_file)



