import argparse
import json
import os
import time
import torch
import psutil
import numpy as np
import asyncio
from tqdm import tqdm
from openai import OpenAI, AsyncOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--ip", type=str, default="10.34.0.228")
    parser.add_argument("--port", type=int, default=6001)
    parser.add_argument("--checkpoint_path", type=str, default="/localhome/local-hndo/hndo/checkpoint/mistral-text2sql-v24")
    # parser.add_argument("--tokenizer_path", type=str, default="/localhome/local-hndo/hndo/tokenizer/sqlcoder-70b-alpha")
    parser.add_argument("--max_seq_length", type=int, default=4096)  # Reduced from 4096 to leave room for DDL
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="/localhome/local-hndo/hndo/sql_evaluation/results-ddl-inst/mistral-text2sql-v24")
    parser.add_argument("--dataset_path", type=str, default="/localhome/local-hndo/hndo/sql_evaluation/dataset/initial_process/test_ehrsql_eicu_data.json")
    parser.add_argument("--metadata_path", type=str, default="/localhome/local-hndo/hndo/sql_evaluation/dataset/metadata/instruct_ddl/eicu_instruct.sql")
    # parser.add_argument("--stop_token", type=str, default="<extra_id_1>")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    task_name = args.task_name
    checkpoint_path = args.checkpoint_path
    model_name = checkpoint_path.split("/")[-1]
    max_seq_length = args.max_seq_length
    batch_size = args.batch_size
    save_dir = args.save_dir
    dataset_path = args.dataset_path
    metadata_path = args.metadata_path
    
    total_tokens = 0
    total_time = 0
    memory_usage = []
    latencies = []
    throughputs = []
    peak_memory = 0
    process = psutil.Process()
    
    api_key = "token-abc123"
    url_base = f"http://{args.ip}:{args.port}/v1"
    # prompt_file = "/localhome/local-hndo/hndo/sql_evaluation/dataset/metadata/prompt.md"
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    client = OpenAI(
        api_key=api_key,
        base_url=url_base,
    )
    
    async_client = AsyncOpenAI(
        api_key=api_key,
        base_url=url_base,
    )

    async def calling_model_async(messages, retries=3, delay=5):
        for attempt in range(retries):
            try:
                response = await async_client.chat.completions.create(
                    model=checkpoint_path,
                    messages=messages,
                    temperature=0.0,
                    top_p=0.95,
                    max_tokens=max_seq_length,
                    stop=["<extra_id_1>"],  # Changed from stop_token_ids to stop
                    timeout=300  # Increased to 5 minutes for slower generation
                )
                return response.choices[0].message.content, response.usage.total_tokens
            except Exception as e:
                print(f"Error: {e}, retrying {attempt+1}/{retries}...")
                await asyncio.sleep(delay)
        raise RuntimeError("Failed after retries")

    def calling_model(messages, retries=3, delay=5):
        for attempt in range(retries):
            try:
                response = client.completions.create(
                    model=checkpoint_path,
                    prompt=messages,
                    temperature=0.0,
                    top_p=0.95,
                    max_tokens=max_seq_length,
                    timeout=60  # Set timeout to 60 seconds
                )
                return response.choices[0].text, response.usage.total_tokens
            except Exception as e:
                print(f"Error: {e}, retrying {attempt+1}/{retries}...")
                time.sleep(delay)
        raise RuntimeError("Failed after retries")
    
    async def process_batch_async(batch_prompt):
        """Process a batch of prompts concurrently"""
        tasks = [calling_model_async(prompt) for prompt in batch_prompt]
        results = await asyncio.gather(*tasks)
        return zip(*results)  # Separate SQL queries and tokens
    
    with open(metadata_path, "r") as f:
        table_metadata_string = f.read()
    
    def generate_prompt(question, ddl, instruction):
        # with open(prompt_file, "r") as f:
        #     prompt = f.read()
        
        # with open(metadata_file, "r") as f:
        #     table_metadata_string = f.read()
            
        # prompt_template = '''Based on DDL statements, instructions, and the current date, generate a SQL query in the following sqlite to answer the question:\n\nDDL statements:\n{ddl}\nInstructions:\n{instruction}\nQuestion:\n{question}.'''
        
        prompt_chat_template = [
            {
                "role": "system",
                "content": f"Today is 2105-12-31 23:59:00. Based on DDL statements, instructions, and the current date, generate a SQL query in the following sqlite to answer the question:\n\nDDL statements:\n{ddl}\nInstructions:\n{instruction}",
            },
            {
                "role": "user",
                "content": f"{question}"
            }
        ]

        # prompt = prompt_chat_template.format(
        #     user_question=question, ddl=table_metadata_string, instruction=instruction
        # )
        
        return prompt_chat_template
    
    save_path = f"{args.save_dir}/{model_name}_{task_name}_result.jsonl"
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            extra_data = [json.loads(x) for x in f.readlines()]
        processed_input_contents = [x["index"] for x in extra_data]
    else:
        extra_data, processed_input_contents = [], []
        
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        # samples = [json.loads(x) for x in f.readlines()]
        samples = json.load(f)
        samples = [x for x in samples if x["index"] not in processed_input_contents]
        
    batched_samples = []
    for i in range(0, len(samples), batch_size):
        batched_samples.append(samples[i:i+batch_size])
        
    for batch_sample in tqdm(batched_samples, total=len(batched_samples)):
        # Record start time and memory
        start_time = time.time()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # batch_prompt = zip(*[generate_prompt(x["user_query"], table_metadata_string, x["instructions"]) for x in batch_sample])
        if task_name == "vinmec":
            batch_prompt = [generate_prompt(x["user_query"], x["ddl"], x["instructions"]) for x in batch_sample]
        else:   
            batch_prompt = [generate_prompt(x["user_query"], table_metadata_string, x["instructions"]) for x in batch_sample]
        # batch_sql_query, batch_tokens = zip(*[calling_model(prompt) for prompt in batch_prompt])
        
        # Use async batch processing for better VLLM utilization    
        batch_sql_query, batch_tokens = asyncio.run(process_batch_async(batch_prompt))
        
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        current_memory = end_memory
        if current_memory > peak_memory:
            peak_memory = current_memory
            
        # Calculate batch metrics
        batch_time = end_time - start_time
        batch_memory = end_memory - start_memory
        # batch_tokens = sum(len(tokenizer.encode(x)) for x in batch_sql_query)
        
        # Update metrics
        total_time += batch_time
        total_tokens += sum(batch_tokens)
        memory_usage.append(batch_memory)
        latencies.append(batch_time * 1000)  # Convert to ms
        throughputs.append(sum(batch_tokens) / batch_time)  # tokens/sec
        
        for sql_query, sample in zip(batch_sql_query, batch_sample):
            # output = output.strip()
            # if output.endswith(stop_token):
            #     output = output[:-len(stop_token)]
                
            # sql_query = output.split("<extra_id_1>Assistant")[-1].strip()
            # if sql_query.startswith(stop_token):
            #     sql_query = sql_query[len(stop_token):].strip()
            # if stop_token in sql_query:
            #     sql_query = sql_query.split(stop_token)[0].strip()
            
            sample.update({"predict": sql_query, "ddl": table_metadata_string})
            extra_data.append(sample)
            
        with open(save_path, 'w', encoding='utf-8') as f:
            lines = [json.dumps(x, ensure_ascii=False) for x in extra_data]
            f.writelines('\n'.join(lines))

        del batch_prompt
        del batch_sql_query

    # Calculate and print final metrics
    avg_memory = np.mean(memory_usage)
    avg_latency = np.mean(latencies)
    avg_throughput = np.mean(throughputs)
    total_throughput = total_tokens / total_time

    print("\nPerformance Metrics:")
    print(f"Average Memory Usage: {avg_memory:.2f} MB")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Average Throughput: {avg_throughput:.2f} tokens/sec")
    print(f"Total Throughput: {total_throughput:.2f} tokens/sec")
    print(f"Total Tokens Processed: {total_tokens}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Peak Memory Usage: {peak_memory:.2f} MB")

    # Save metrics to file
    metrics = {
        "average_memory_mb": avg_memory,
        "peak_memory_mb": peak_memory,
        "average_latency_ms": avg_latency,
        "average_throughput_tokens_per_sec": avg_throughput,
        "total_throughput_tokens_per_sec": total_throughput,
        "total_tokens": total_tokens,
        "total_time_sec": total_time,
        "batch_size": batch_size,
        "num_batches": len(batched_samples)
    }
    save_metric_path = f"{args.save_dir}/{model_name}_{task_name}_batch_{args.batch_size}_system_metrics.json"
    with open(save_metric_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

        
    