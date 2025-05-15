import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModel, MistralForCausalLM
from transformers import GenerationConfig, LogitsProcessorList
from transformers import LlamaForCausalLM, LlamaTokenizer
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
import sys
import random
from watermark_processor.extended_watermark_processor import WatermarkLogitsProcessor



TEMPLATE_LLAMA2 = (
    "[INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n"
    "{instruction} [/INST]"
)

TEMPLATE_LLAMA3 = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    "{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
)

TEMPLATE_MISTRAL = (
    "[INST] {system_prompt} {instruction} [/INST]"
)


DEFAULT_SYSTEM_PROMPT = """ you are a helpful assistant. """

def generate_prompt(args, instruction, system_prompt=DEFAULT_SYSTEM_PROMPT):
    if args.model_name == "llama2":
        return TEMPLATE_LLAMA2.format_map({'instruction': instruction,'system_prompt': system_prompt})
    elif args.model_name == "llama3":
        return TEMPLATE_LLAMA3.format_map({'instruction': instruction,'system_prompt': system_prompt})
    elif args.model_name == "mistral":
        return TEMPLATE_MISTRAL.format_map({'instruction': instruction,'system_prompt': system_prompt})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tuning')
    parser.add_argument('--wm', type=float, default=1, help="watermart ratio")
    parser.add_argument('--model_name', type=str, default="llama2", help="model name")
    parser.add_argument('--delta', type=float, default=3.0)
    parser.add_argument('--hash_key_idx', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="code", help="dataset")

    args = parser.parse_args()

    wm = args.wm
    begin_idx = 0
    data_length = 5000
    delta = args.delta


    if args.model_name == "llama2":
        TARGET_MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"
    elif args.model_name == "llama3":
        TARGET_MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif args.model_name == "mistral":
        TARGET_MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.2"


    device = 'cuda'

    if args.dataset == "code":
        data_path = "./Evol-Instruct-Code-80k-v1/EvolInstruct-Code-80k.json"
        with open(data_path) as f:
            data = json.load(f)
        query_list = [reg['instruction'] for reg in data]

    elif args.dataset == "math":
        data_path = "./gsm8k/train.json"
        with open(data_path) as f:
            data = json.load(f)
        query_list = [reg['question'] for reg in data]

    if wm == 0:
        OUTPUT_DIR = "./data_for_distill/{}_{}_no_wm.json".format(args.model_name, args.dataset)
    else:
        OUTPUT_DIR = "./data_for_distill/{}_{}_wm_{}_delta_{}.json".format(args.model_name, args.dataset, args.hash_key_idx, args.delta)

    random.seed(776)
    hash_key_list = [random.randint(0, 2**31 - 1) for _ in range(10)]


    np.random.seed(20)
    torch.manual_seed(20)
    torch.cuda.manual_seed_all(20)


    # initialize model
    model = AutoModelForCausalLM.from_pretrained(
                        TARGET_MODEL_PATH,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
                        TARGET_MODEL_PATH,
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
    watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                            gamma=0.25,
                                            delta=args.delta,
                                            new_hash_key=hash_key_list[args.hash_key_idx],
                                            seeding_scheme="static")
    
    if os.path.exists(OUTPUT_DIR): 
        with open(OUTPUT_DIR, "r") as f:
            output_list = json.load(f)
        begin_idx = len(output_list['input'])
        data_length = 5000 - begin_idx
    else:
        output_list = {'input':[], 'output':[]}
    for query_idx in tqdm(range(begin_idx, begin_idx+data_length), total=data_length):
        
        input_text = generate_prompt(args, instruction=query_list[query_idx], system_prompt=DEFAULT_SYSTEM_PROMPT)
        inputs = tokenizer(input_text, return_tensors="pt")
        if wm==0:
            generation_output = model.generate(
                input_ids = inputs["input_ids"].to(device),
                attention_mask = inputs['attention_mask'].to(device),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                generation_config = generation_config
            )
        else:
            
            generation_output = model.generate(
                input_ids = inputs["input_ids"].to(device),
                attention_mask = inputs['attention_mask'].to(device),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                generation_config = generation_config,
                logits_processor=LogitsProcessorList([watermark_processor])
            )

        s = generation_output[0]
        
        if args.model_name == "llama2":
            output = tokenizer.decode(s,skip_special_tokens=True)
            response = output.split("[/INST]")[-1].strip()
        elif args.model_name == "llama3":
            output = tokenizer.decode(s,skip_special_tokens=False)
            response = output.split("assistant<|end_header_id|>")[1].split("<|eot_id|>")[0].strip()
        elif args.model_name == "mistral":
            output = tokenizer.decode(s,skip_special_tokens=True)
            response = output.split("[/INST]")[-1].strip()
        if query_idx==0:
            print(query_list[query_idx])
            print(response)
        output_list['input'].append(query_list[query_idx])
        output_list['output'].append(response)
    save_file = json.dumps(output_list)
    f = open(OUTPUT_DIR, 'w')
    f.write(save_file)



