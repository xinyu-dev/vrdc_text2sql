import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=8192) 
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--dataset_path", type=str, default="/home/jovyan/lustre/users/hvnguyen/data/nemo/text2sql/deepseek_r1_vinmec_generated_data/mistral_minitron_test.jsonl")
    parser.add_argument("--stop_token", type=str, default="<extra_id_1>")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    task_name = args.task_name
    max_seq_length = args.max_seq_length

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        local_files_only=True, 
        trust_remote_code=True, 
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
        use_fast=False,
        max_length=max_seq_length
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"  # Automatically assigns devices
    )

    # process save_path and cache
    save_path = f"{args.save_dir}/text_{task_name}_result.jsonl"
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            extra_data = [json.loads(x) for x in f.readlines()]
        processed_input_contents = [x["input"] for x in extra_data]
    else:
        extra_data, processed_input_contents = [], []

    # load datasets
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        samples = [json.loads(x) for x in f.readlines()]
        samples = [x for x in samples if x["input"] not in processed_input_contents]

    # group data to batch
    batched_samples = []
    batch_size = args.batch_size
    for i in range(0, len(samples), batch_size):
        batched_samples.append(samples[i:i+batch_size])

    # run inference
    stop_token = args.stop_token
    for batch_sample in tqdm(batched_samples, total=len(batched_samples)):
        batch_input = [x["input"] for x in batch_sample]
        batch_chat_prompt = tokenizer(
            batch_input, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length
        ).to("cuda")
        
        outputs = model.generate(
            **batch_chat_prompt, 
            stop_strings=[stop_token], 
            tokenizer=tokenizer,
            max_length=max_seq_length,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,   
        )
        outputs = tokenizer.batch_decode(outputs)
        for output, sample in zip(outputs, batch_sample):
            output = output.strip()
            if output.endswith(stop_token):
                output = output[:-len(stop_token)]
                
            sql_query = output.split("<extra_id_1>Assistant")[-1].strip()
            if sql_query.startswith(stop_token):
                sql_query = sql_query[len(stop_token):].strip()
            if stop_token in sql_query:
                sql_query = sql_query.split(stop_token)[0].strip()
            sample.update({"predict": sql_query})
            extra_data.append(sample)
            
        with open(save_path, 'w', encoding='utf-8') as f:
            lines = [json.dumps(x, ensure_ascii=False) for x in extra_data]
            f.writelines('\n'.join(lines))

        del batch_chat_prompt
        del outputs


