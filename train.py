import os, sys
import torch
import datasets
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    GenerationConfig,
    AutoModel
)
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
import json
from datasets import Dataset
import argparse
import bitsandbytes as bnb 

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
    if param.requires_grad:
        trainable_model_params += param.numel()
    print(f"all params num: {all_model_params}, trainable param num: {trainable_model_params}")
    return trainable_model_params

def generate_prompt(instruction, label=None, system_prompt=DEFAULT_SYSTEM_PROMPT):
    res = prompt_template["prompt_input"].format(instruction=instruction, system_prompt=DEFAULT_SYSTEM_PROMPT)
    if label:
        res = "{res} {label}"
    return res

def generate_train_data(data_path, data_length=-1):
    train_data = []
    with open(data_path) as f:
        data = json.load(f)
    if data_length == -1:
        data_length = len(data['input'])
    for data_idx in range(data_length):
        data_dict = {'instruction': prompt_template["prompt_input"].format(instruction=data['input'][data_idx], system_prompt=DEFAULT_SYSTEM_PROMPT), 'response': data['output'][data_idx]+" </s>"}
        train_data.append(data_dict)
    return train_data


def tokenize(data):
    source_ids = tokenizer.encode(data['instruction'])
    target_ids = tokenizer.encode(data['response'])
    input_ids = source_ids + target_ids + [tokenizer.eos_token_id]
    labels = [-100] * len(source_ids) + target_ids + [tokenizer.eos_token_id]
    return {
        "input_ids": input_ids,
        "labels": labels
    }
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tuning')
    parser.add_argument('--wm', type=float, default=1, help="watermart ratio")
    parser.add_argument('--delta', type=float, default=3.0)
    parser.add_argument('--source_name', type=str, default="llama2", help="model name")
    parser.add_argument('--target_name', type=str, default="bloom", help="model name")
    parser.add_argument('--hash_key_idx', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="code", help="dataset")
    parser.add_argument('--data_length', type=int, default=-1)
    args = parser.parse_args()

    wm = args.wm
    target_name = args.target_name
    if target_name == "mistral":
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    elif target_name == "bloom":
        model_id = "bigscience/bloom-7b1"

    device_map = "auto"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,   # load the model into memory using 4-bit precision
        bnb_4bit_use_double_quant=True, # use double quantition
        bnb_4bit_quant_type="nf4", # use NormalFloat quantition
        bnb_4bit_compute_dtype=torch.bfloat16 # use hf for computing when we need
    )


    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        use_cache=False,
        trust_remote_code=True,
        device_map=device_map
    )



    tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ori_p = print_number_of_trainable_model_parameters(model)

    model = prepare_model_for_kbit_training(model)


    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        # lora_dropout=0.1,
        # target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )


    model = get_peft_model(model, peft_config)

    ### compare trainable parameters
    peft_p = print_number_of_trainable_model_parameters(model)

    print(f'# Trainable parameter \nBefore: {ori_p}\nAfter: {peft_p} \nPercentage: {round(peft_p/ori_p * 100, 2)}')


    prompt_template = {
        "prompt_input": "[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction} [/INST]"
    }

    DEFAULT_SYSTEM_PROMPT = """ you are a helpful assistant. """




    if wm == 0:
        data_path = "./data_for_distill/{}_{}_no_wm.json".format(args.source_name, args.dataset)
    else:
        data_path = "./data_for_distill/{}_{}_wm_{}_delta_{}.json".format(args.source_name, args.dataset, args.hash_key_idx, args.delta)



    data_length = args.data_length

    data = generate_train_data(data_path, data_length=data_length)
    data_dict = {key: [dic[key] for dic in data] for key in data[0]}

    dataset = Dataset.from_dict({key: [dic[key] for dic in data] for key in data[0]})
    cols = ["instruction", "response"]
    train_data = dataset.map(tokenize)
    print(train_data)

    if wm==0:
        model_save_path = "./distilled_models/{}_from_{}_{}_no_wm.json".format(args.target_name, args.source_name, args.dataset)
    else:
        if data_length==-1:
            model_save_path = "./distilled_models/{}_from_{}_{}_wm_{}_delta_{}.json".format(args.target_name, args.source_name, args.dataset, args.hash_key_idx, args.delta)
        else:
            model_save_path = "./distilled_models/{}_from_{}_{}_wm_{}_delta_{}.json".format(args.target_name, args.source_name, args.dataset+str(data_length), args.hash_key_idx, args.delta)


    args = TrainingArguments(
        report_to="none",
        output_dir=model_save_path,
        num_train_epochs=4,
        fp16=True,
        optim="paged_adamw_32bit",
        learning_rate=1e-4,
        lr_scheduler_type="constant",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        group_by_length=False,
        save_strategy="steps",
        save_steps=200,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        args=args,
        data_collator=DataCollatorForSeq2Seq(
        tokenizer, 
        pad_to_multiple_of=8, 
        return_tensors="pt", 
        padding=True
        ),
    )

    # silence the warnings. re-enable for inference!
    model.config.use_cache = False
    IS_RESUME = False

    if IS_RESUME:
        trainer.train(f'{model_save_path}/checkpoint-200')
    else:
        trainer.train()
        model.save_pretrained(model_save_path)
        print('model train is finished')