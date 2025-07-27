# Model Evaluation Framework

A comprehensive framework for evaluating text-to-SQL models with support for multiple inference methods and evaluation datasets.

## 📋 Table of Contents

1. [Model Converter](#1-convert-model-to-huggingface    )
2. [Inference Methods](#2-inference-methods)
3. [Evaluation](#3-evaluation)

---

## 1. Convert model to HuggingFace
Model checkpoints are finetuned with NeMo Framework (version 1.0) which save model in format .nemo file. So you need to convert these checkpoints to Huggingface format. See example about nvidia/Mistral-NeMo-Minitron-8B-Instruct model:

- Step 1: Save your checkpoint to checkpoints folder. Then download base model from huggingface and save to base_model folder. You can download it by huggingface-hub

```
mkdir base_models/Mistral-NeMo-Minitron-8B-Instruct
huggingface-cli download nvidia/Mistral-NeMo-Minitron-8B-Instruct --local-dir base_models/Mistral-NeMo-Minitron-8B-Instruct
```

- Step 2: The converter code with be run in NeMo Docker container with version 24.07

```
mkdir hf_checkpoints
cd model_converter
docker build . -t nemo_converter:v1.0
docker run \
    --name NeMoConverterService -it --gpus all \
    -v ../base_models:/workspace/base_models \
    -v ../checkpoints:/workspace/checkpoints \
    -v ../hf_checkpoints:/workspace/hf_checkpoints \
    nemo_converter:v1.0
```

- Step 3: Attach to the container and convert model

```
docker exec -it NeMoConverterService bash
python workspace/convert_mistral_2_hf.py \
    --input_name_or_path /workspace/checkpoints/your-checkpoint \
    --output_path /workspace/hf_checkpoints/Finetuned-Mistral-NeMo-Minitron-8B-Instruct \
    --hf_model_name /workspace/base_models/Mistral-NeMo-Minitron-8B-Instruct
```

---

## 2. Inference Methods

There are three different inference approaches based on your needs:

### Method 1: Direct Inference (HuggingFace Transformers)

**Script:** `model_inference/inference_scripts/mistral_infer.py`

### Method 2: vLLM Endpoint Deployment

**Script:** `model_inference/inference_scripts/mistral-vllm.py`

**Start vLLM Server:**
```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve <path_or_hf_repo_to_model> \
    --dtype auto \
    --api-key <API_KEY> \
    --port 8000 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --max-num-seqs 8 \
    --tensor-parallel-size 2 
```

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /home/ubuntu/workspace/mistral-nemo-minitron-8b-instruct-healthcare-text2sql_vV2.8 \
    --dtype auto \
    --port 8000 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --tensor-parallel-size 1
```


**Parameter Explanations:**
- `<path_or_hf_repo_to_model>`: Path to your local model or HuggingFace repo name 
- `--api-key <API_KEY>`: Authentication key for accessing the endpoint
- `--port 8000`: Port number for the server 
- `CUDA_VISIBLE_DEVICES=0,1`: Specify which GPUs to use 
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism (must match CUDA_VISIBLE_DEVICES)
- `--gpu-memory-utilization`: Fraction of GPU memory to use (0.0-1.0)
- `--dtype auto`: Data type for model weights (auto, float16, bfloat16, float32)
- `--max-model-len`: Maximum sequence length the model can handle
- `--max-num-seqs`: Maximum number of concurrent sequences to process

### Method 3: RAG-Enhanced Inference
We implemented RAG by embedding all training examples and using similarity search to retrieve relevant context. The process involves:

1. **Embedding Generation:** Convert all training questions into embeddings using hte model `nvidia/nv-embedqa-mistral-7b-v2`
2. **Vector Index Construction:** Build a FAISS-based vector database to enable fast similarity search.
3. **Query Retrieval:** For each test question, retrieve the top k most similar training examples (here, we use k = 3).
4. **Context Integration:** Incorporate the retrieved examples into the prompt to improve the model’s response quality.

**Example script with vLLM endpoint:** `model_inference/rag/mistral-text2sql-rag-vllm.py`

#### Setup Prerequisites

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Environment Configuration**
Create `.env` file in project root:
```
NVIDIA_API_KEY=your_nvidia_api_key_here
```

**3. Training Data Preparation**
Create CSV file with `questions` and `labels` columns. See example format in `model_inference/rag/ehrsql_eicu/train_val.csv`.

**4. Create Database Embedding Cache File (.pkl)**

The embedding cache file (e.g., `train_database_nv_embedcode_7b.pkl`) is required for RAG inference.

**Key Functions in `model_inference/rag/rag.py`:**
- `create_faiss_index()`: Main function to create .pkl cache file
- `generate_database_embeddings()`: Handles embedding generation and caching
- `QueryData`: Class to structure training data

**Note:** The .pkl file will be used in RAG inference via the `--embedding_cache_file` parameter.

#### RAG Inference Execution

**1. Start vLLM Server** \
Detail in method 2.

**2. Run RAG Inference**
```bash
python model_inference/rag/mistral-text2sql-rag-vllm.py \
    --task_name ehrsql_eicu \
    --ip localhost \
    --port 6001 \
    --checkpoint_path /path/to/your/model \
    --max_seq_length 4096 \
    --batch_size 8 \
    --save_dir model_evaluation/model_predictions/mistral-text2sql-v282 \
    --train_file rag/ehrsql_eicu/train_val.csv \
    --embedding_model_name nvidia/nv-embedqa-mistral-7b-v2 \
    --embedding_cache_file rag/ehrsql_eicu/train_database_nv_embedcode_7b.pkl \
    --top_k 3 \
    --dataset_path model_evaluation/dataset/test/test_ehrsql_eicu_data.json \
    --metadata_path model_evaluation/dataset/metadata/eicu_instruct.sql
```

**Key Parameters:**
- `--task_name`: Dataset name (e.g., ehrsql_eicu, mimicsql)
- `--ip` and `--port`: vLLM server connection details
- `--checkpoint_path`: Path to your fine-tuned model
- `--train_file`: CSV file with training examples for RAG retrieval
- `--embedding_model_name`: NVIDIA embedding model for vector search
- `--embedding_cache_file`: Cache file for pre-computed embeddings
- `--top_k`: Number of similar examples to retrieve (default: 3)
- `--dataset_path`: Test dataset JSON file
- `--metadata_path`: Database schema/metadata file

**Outputs:**
- Model predictions in JSONL format
- System performance metrics (memory, latency, throughput)
- FAISS index cache for faster subsequent runs

**Performance Optimization:**
- Adjust `--batch_size` based on GPU memory
- Use `--top_k` to control retrieval context size
- Monitor memory usage and adjust `--max-num-seqs` accordingly

---

## 3. Evaluation

Comprehensive evaluation scripts for both SQLite and PostgreSQL databases.

### 3.1 SQLite Database Evaluation

**Supported Datasets:** MIMICSQL & EHRSQL \
**Scripts:** 
* `ehrsql_eval.py` (EHRSQL)
* `mimicsql_eval.py` (MIMICSQL)

#### Arguments
- `--pred_file`: Path to model predictions (JSON/JSONL)
- `--db_path`: Path to SQLite database file
- `--num_workers`: Parallel workers (default: -1 for all CPUs)
- `--timeout`: Query execution timeout (seconds)
- `--out_file`: Output directory for results and metrics
- `--ndigits`: Number of digits to round scores (default: 2)

#### Example Commands

**Direct Execution:**
```bash
cd model_evaluation
python ehrsql_eval.py \
    --pred_file model_predictions/test_mistral-text2sql-ehrsql-eicu_result.jsonl \
    --db_path databases/eicu.sqlite \
    --num_workers -1 \
    --timeout 60 \
    --out_file outputs \
    --ndigits 2
```

**Using Helper Script:**
```bash
bash scripts/ehrsql_eval.sh
```

#### Outputs
- Per-sample evaluation results (JSONL)
- Overall metrics (precision, recall, F1, accuracy) in JSON
- Error logs for failed queries

### 3.2 PostgreSQL Database Evaluation

**Supported Dataset:** Vinbrain \
**Script:** `psql_sql_eval.py`

#### Configuration
**Config File:** `configs/sql_eval.yaml`
- Database connection information
- Error logging template

#### Arguments
- `--config`: Path to YAML config (default: configs/sql_eval.yaml)
- `--pred_file`: Path to model predictions (JSON/JSONL)
- `--out_file`: Output directory for results and metrics
- `--input_template`: Input format ('chat' or 'string')

#### Example Commands

**Direct Execution:**
```bash
cd model_evaluation
python psql_sql_eval.py \
    --config configs/sql_eval.yaml \
    --pred_file model_predictions/model_base/mistral-nemo-minitron-8b-instruct/mistral-nemo-minitron-8b-instruct_vinmec_results.json \
    --out_file results \
    --input_template string
```

**Using Helper Script:**
```bash
bash scripts/psql_sql_eval.sh
```

#### Outputs
- Per-sample evaluation results (JSONL)
- Overall metrics (exact match, subset match, run errors) in JSON
- Error logs for failed queries
