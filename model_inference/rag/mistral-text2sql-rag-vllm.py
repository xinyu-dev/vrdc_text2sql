import argparse
import json
import os, sys
import time
import torch
import psutil
import numpy as np
import pandas as pd
import asyncio
from tqdm import tqdm
from openai import OpenAI, AsyncOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from rag import create_faiss_index, find_similar, QueryData
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# NOTE: ===== Experimental =====
sys.path.append('/root/workspace/vrdc_text2sql/model_evaluation')
from utils.experimental import FAISSRetriever, split_sql_blocks
# NOTE: ===== End of Experimental =====


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=6001)
    parser.add_argument("--checkpoint_path", type=str, default="/home/jovyan/lustre/users/hvnguyen/experiments/nemo_1.0/mistral-nemo-minitron-8b-instruct-text2sql-dv8.2-pv2-24x4/checkpoints/hf")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="results/vllm/mistral-text2sql-v282")
    # retrieval
    parser.add_argument("--train_file", default="/home/jovyan/lustre/users/hndo/rag_text2sql/ehrsql_eicu/train_val.csv")
    parser.add_argument("--embedding_model_name", default="nvidia/nv-embedqa-mistral-7b-v2")
    parser.add_argument("--embedding_cache_file", default="/home/jovyan/lustre/users/hndo/rag_text2sql/ehrsql_eicu/train_database_nv_embedcode_7b.pkl")
    parser.add_argument("--top_k", type=int, default=3)
    
    parser.add_argument("--dataset_path", type=str, default="/home/jovyan/lustre/users/hndo/text2sql_eval/benchmark/processed_data/initial/test_ehrsql_eicu_data.json")
    parser.add_argument("--metadata_path", type=str, default="/home/jovyan/lustre/users/hndo/text2sql_eval/benchmark/processed_data/metadata/eicu_instruct.sql")
    parser.add_argument("--format", type=str, default="sqlite")
    parser.add_argument("--experimental", action="store_true") # enable experimental features such as DDL retrieval to increase accuracy
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
    format_sql = args.format
    
    total_tokens = 0
    total_time = 0
    memory_usage = []
    latencies = []
    throughputs = []
    peak_memory = 0
    process = psutil.Process()
    
    # ===== Client for text2SQL model =====
    # For finetuned Mistral model: 
    async_client = AsyncOpenAI(
        api_key=os.getenv("NGC_API_KEY"),
        base_url=f"http://{args.ip}:{args.port}/v1",
    )

    # For claude: 
    # async_client = AsyncOpenAI(
    #     api_key=os.environ['BEDROCK_OPENAI_API_KEY'],
    #     base_url=os.environ['BEDROCK_OPENAI_BASE_URL']
    # )
    # ===== End of client for text2SQL model =====

    if not args.experimental:
        logger.info("Experimental features are disabled.")

    # NOTE: ===== experimental =====
    if args.experimental:
        logger.warning("Using experimental features...")
        # Chunk and retrieve DDL blocks
        # read ddl from file
        sql_file = args.metadata_path
        assert os.path.exists(sql_file), f"SQL file {sql_file} does not exist"

        # Split the SQL file into blocks
        blocks = split_sql_blocks(sql_file)
        logger.info(f"Number of DDL blocks: {len(blocks)}")

        # create a retriever. Must use OpenAI embedding due to length limit of Mistral embedding model
        # ddl_retreiver = FAISSRetriever(
        #     api_key=os.getenv("LLM_GATEWAY_API"),
        #     api_version=os.getenv("LLM_GATEWAY_API_VERSION"),
        #     endpoint = os.getenv("LLM_GATEWAY_ENDPOINT"),
        #     model = "text-embedding-3-large", 
        # )

        ddl_retreiver = FAISSRetriever(
            api_key=os.getenv("NGC_API_KEY"),
            endpoint = "https://integrate.api.nvidia.com/v1",
            model = "nvdev/nvidia/llama-3.2-nv-embedqa-1b-v2",  # remove the "nvdev/" if using public playground 
        )

        # specify the cache file for the DDL vector database. Use None to re-generate the embeddings without saving. 
        # ddl_embedding_cache_file = "/root/workspace/vrdc_text2sql/model_evaluation/dataset/train_eval/eicu/ddl_database_openai_text-embedding-3-large.pkl"
        ddl_embedding_cache_file = "/root/workspace/vrdc_text2sql/model_evaluation/dataset/train_eval/eicu/ddl_database_llama-3.2-nv-embedqa-1b-v1.pkl"
        ddl_retreiver.embed_blocks(
            text_blocks=blocks,
            cache_file=ddl_embedding_cache_file,
        )
    # ===== end of experimental =====
    
    
    async def calling_model_async(messages, retries=3, delay=5):
        for attempt in range(retries):
            try:
                # For finetuned Mistral model: 
                # response = await async_client.chat.completions.create(
                #     model=checkpoint_path,
                #     messages=messages,
                #     temperature=0.0,
                #     top_p=0.95,
                #     max_tokens=max_seq_length,
                #     stop=["<extra_id_1>"],  # Changed from stop_token_ids to stop
                #     timeout=300  # Increased to 5 minutes for slower generation
                # )

                # for claude: 
                response = await async_client.chat.completions.create(
                    model=checkpoint_path,
                    messages=messages,
                    temperature=0.0,
                    top_p=0.95,
                    max_completion_tokens=max_seq_length,
                )
                return response.choices[0].message.content, response.usage.total_tokens
            except Exception as e:
                print(f"Error: {e}, retrying {attempt+1}/{retries}...")
                await asyncio.sleep(delay)
        raise RuntimeError("Failed after retries")
    
    async def process_batch_async(batch_prompt):
        """Process a batch of prompts concurrently"""
        tasks = [calling_model_async(prompt) for prompt in batch_prompt]
        results = await asyncio.gather(*tasks)
        return zip(*results)  # Separate SQL queries and tokens
    
    with open(metadata_path, "r") as f:
        table_metadata_string = f.read()
    
    def generate_prompt(question, ddl, instruction, retrieved_ids, train_pd, current_date, format_sql):

        # retrieve Q&A blocks
        retrieved_samples = "Here are some sample question and correct SQL (may or may not be useful in constructing the SQL to answer the question) :\n\n"
        for idx in retrieved_ids:
            q = str(train_pd.loc[idx, "questions"])
            sql = str(train_pd.loc[idx, "labels"])
            if sql.lower() == "nan":  # missing label
                retrieved_samples += f"Question: {q}.  SQL answer: \nNone\n"
            else:
                retrieved_samples += f"Question: {q}.  SQL answer: ```sql\n{sql}\n```\n"

        # NOTE:===== experimental =====
        if args.experimental:
            # try retriving relevant blocks from the DDL vector database
            retrieved_blocks = ddl_retreiver.retrieve(
                query = question,
                top_k=5 # top_k for DDL chunk retrieval
            )
            ddl = "\n".join([b['content'] for b in retrieved_blocks])
        # NOTE: ===== end of experimental =====

        prompt_chat_template_rag = [
            {
                "role": "system",
                "content": f"Based on DDL statements, instructions, and the current date, generate a SQL query in the following {format_sql} to answer the question.\nIf the question cannot be answered using the available tables and columns in the DDL (i.e., it is out of scope), return only: None.\nToday is {current_date}\nDDL statements:\n{ddl}\n{retrieved_samples}\nInstructions:\n{instruction}",
            },
            {
                "role": "user",
                "content": f"{question}"
            }
        ]
        
        return prompt_chat_template_rag
    
    save_path = f"{args.save_dir}/test_rag_vllm_{args.task_name}_result_mis_embedd.jsonl"
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            extra_data = [json.loads(x) for x in f.readlines()]
        processed_input_contents = [x["index"] for x in extra_data]
    else:
        extra_data, processed_input_contents = [], []

    # ---------------------------------------------------------------------
    # Load prompts, training data, and build / load FAISS index
    # ---------------------------------------------------------------------
    train_pd = pd.read_csv(args.train_file)

    database = [
        QueryData(i, row["questions"], row["labels"]) for i, row in train_pd.iterrows()
    ]
    database_index = create_faiss_index(database, args.embedding_model_name, args.embedding_cache_file)
    
    # ---------------------------------------------------------------------
    # Load test dataset
    # ---------------------------------------------------------------------
        
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

        batch_prompt = []
        for x in batch_sample:
            retrieved_ids = find_similar(
                x["user_query"], database, database_index, args.embedding_model_name, top_k=args.top_k
            )

            prompt = generate_prompt(x["user_query"], table_metadata_string, x["instructions"], retrieved_ids, train_pd, x["current_date"], format_sql)
            batch_prompt.append(prompt)
                 
        batch_sql_query, batch_tokens = asyncio.run(process_batch_async(batch_prompt))
        
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        current_memory = end_memory
        if current_memory > peak_memory:
            peak_memory = current_memory
            
        # Calculate batch metrics
        batch_time = end_time - start_time
        batch_memory = end_memory - start_memory
        
        # Update metrics
        total_time += batch_time
        total_tokens += sum(batch_tokens)
        memory_usage.append(batch_memory)
        latencies.append(batch_time * 1000)  # Convert to ms
        throughputs.append(sum(batch_tokens) / batch_time)  # tokens/sec
        
        for sql_query, sample, prompt in zip(batch_sql_query, batch_sample, batch_prompt):
            sample.pop("ddl", None)
            sample.update({"input": json.dumps(prompt, ensure_ascii=False), "predict": sql_query})
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
    save_metric_path = f"{args.save_dir}/{model_name}_{task_name}_batch_{args.batch_size}_system_metrics_mis_embedd.json"
    with open(save_metric_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

        
    