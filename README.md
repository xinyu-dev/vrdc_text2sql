# Text2SQL with RAG


# Setup

## Resources

> [!NOTE]
    > 1. The finetuned mistral-nemo-minitron-8b-instruct-healthcare-text2sql cab be run on a single L40S or A100. A100 is recommended. See [GPU recommendation](https://docs.nvidia.com/nim/large-language-models/latest/supported-models.html#mistral-nemo-minitron-8b-8k-instruct).
    > 2. If you're Claude or OpenAI, a small GPU instance such as A10G is needed in order the faiss GPU cuVS implementation.
    > 3. In my testing, I use 1x A100 80G GPU

## NGC

1. Install NGC CLI:
	```bash
	cd ~ && wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.164.0/files/ngccli_linux.zip -O ngccli_linux.zip && unzip ngccli_linux.zip
	```
2.  Run the following to add executable to path:
	```bash
	chmod u+x ngc-cli/ngc && echo "export PATH=\"\$PATH:$(pwd)/ngc-cli\"" >> ~/.bashrc && source ~/.bashrc
	```
3. Configure NGC:
	```bash
	ngc config set
	```
	- Enter the NGC API key.
	- For org, choose any valid org.
	- For format, type `json` and enter.
	- Everything else leave default.

## Miniconda

> [!NOTE]
    > We recommend using miniconda to manage the environment if you plan to use faiss for RAG. The faiss-GPU-cuVS package is easier to install with conda. If you're using other vectors store, then you can also `uv` or package manager to create environment

1. Run
	```bash
	curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	```
2. Then
	```bash
	bash ~/Miniconda3-latest-Linux-x86_64.sh
	```
3. Click through... Accept default, select `yes` for all questions.
4. Reopen a terminal to take effect. Alternatively, in the same terminal, run run
	```bash
	source ~/.bashrc
	```

## Workspace

1. Create worksplace
	```bash
	mkdir ~/workspace && cd ~/workspace
	```
2. Create uv environment. See [vllm docs](https://docs.vllm.ai/en/stable/getting_started/quickstart.html#prerequisites)
	```bash
	uv venv --python 3.12
	conda create -n workspace python=3.12
	```
3. Activate uv environment
	```bash
	source .venv/bin/activate
	conda activate workspace
	```
4. Install faiss-gpu-cuvs. See [here](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md#installing-faiss-via-conda)
	```bash
	conda install -c pytorch -c nvidia -c rapidsai -c conda-forge libnvjitlink faiss-gpu-cuvs=1.11.0
	```
5. Install vllm. Currently I am using `v0.9.2`
```bash	
pip install vllm "numpy<2" pandas sqlalchemy func-timeout loguru python-dotenv ipykernel
```
6. 

## Model Weights

1. Go to page [here](https://registry.ngc.nvidia.com/orgs/0739904176229748/teams/slm_healthcare/models/mistral-nemo-minitron-8b-instruct-healthcare-text2sql/version). Download model weights
	```bash
	ngc registry model download-version "0739904176229748/slm_healthcare/mistral-nemo-minitron-8b-healthcare-text2sql:1.0"
	```

## Code repo

1. Clone github repo for the inference examples
	```bash
	git clone https://github.com/xinyu-dev/vrdc_text2sql.git
	```

## Example SQLite database (eICU) used in the example

1. CD into the project folder
	```bash
	cd vrdc_text2sql 
	```
2. First create the databases folder
	```bash
	mkdir -p vrdc_text2sql/model_evaluation/databases
	```
3. Download the SQLite example database
	```bash
nngc registry resource download-version "0739904176229748/eicu.sqlite:1.0"
	```
4. This creates a folder `eicu.sqlite_v1.0`. Inside the folder, there is a file `eicu.sqlite`.
5. We will move this file to `databases` folder:
	```bash
	mv eicu.sqlite_v1.0/eicu.sqlite vrdc_text2sql/model_evaluation/databases/ && rmdir eicu.sqlite_v1.0
	```

## Example embedding files used in the example

1. CD into the project folder
	```bash
	cd vrdc_text2sql 
	```
2. First make a directory:
	```
	mkdir -p model_evaluation/dataset/train_eval/eicu
	```
3. Download the cached embeddings:
	```bash
	ngc registry resource download-version "0739904176229748/slm_healthcare/train_eval_eicu:1.0"
	```
4. Move the files to the correct folder:
	```bash
	mv train_eval_eicu_v1.0/* model_evaluation/dataset/train_eval/eicu/ && rmdir train_eval_eicu_v1.0
	```

## Environment Variable

Inside `vrdc_text2sql` folder, create a file called `.env`. Fill out the the environment variables:

```bash
# vrdc_text2sql/.env

# Bedrock Deploy for Claude
# See https://github.com/aws-samples/bedrock-access-gateway on converting Bedrock to OpenAI client
BEDROCK_OPENAI_BASE_URL=
BEDROCK_OPENAI_API_KEY=

# Azure OpenAI Gateway for OpenAI model, for example: 
LLM_GATEWAY_API=
LLM_GATEWAY_API_VERSION=2025-04-01-preview
LLM_GATEWAY_ENDPOINT=https://prod.api.nvidia.com/llm/v1/azure

# NGC API key, used for embedding models from build.nvidia.com
NGC_API_KEY=
```

Then rename the file to `.env`

## Launch server

> [!NOTE]
    > Skip this section if you're using Claude or OpenAI

> [!NOTE]
    > The example code includes FAISS-GPU-CuVS for fast embedding search. This will compete for GPU memory with the vLLM deployed model. If you only have 1 GPU, you might want to reduce the `--gpu-memory-utilization` so that it doesn't generate a memory allocation issue, such as 0.8. If you can multiple GPUs, you can specify the GPU for vLLM deployment to `CUDA_VISIBLE_DEVICES=0` and specify the GPU for FAISS-GPU-CuVS to device 1. See [[#Step 4 Configure DDL embedding model]] .
    >

Make sure the UV environment with vLLM installed is activated. Then run:

Example on brev.dev cloud:
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /home/ubuntu/workspace/mistral-nemo-minitron-8b-healthcare-text2sql_v1.0 \
    --dtype auto \
    --port 8000 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --tensor-parallel-size 1
```

Example on Lepton cluster:
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /root/workspace/mistral-nemo-minitron-8b-healthcare-text2sql_v1.0 \
    --dtype auto \
    --port 8000 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --tensor-parallel-size 1
```

Notes:
- Replace `/home/ubuntu/workspace/mistral-nemo-minitron-8b-instruct-healthcare-text2sql_vV2.8` with the folder that has the model weights downloaded from NGC
- `--api-key <API_KEY>`: Authentication key for accessing the endpoint
- `--port 8000`: Port number for the server
- `CUDA_VISIBLE_DEVICES=0,1,....`: Specify which GPUs to use. For a single L40S instance, we only have 1 GPU, so we set `CUDA_VISIBLE_DEVICES=0 `
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism (must match CUDA_VISIBLE_DEVICES)
- `--gpu-memory-utilization`: Fraction of GPU memory to use (0.0-1.0)
- `--dtype auto`: Data type for model weights (auto, float16, bfloat16, float32)
- `--max-model-len`: Maximum sequence length the model can handle. Use `4` if running on L40S.
- `--max-num-seqs`: Maximum number of concurrent sequences to process

Wait until it says:

```bash
INFO:     Application startup complete.
```

## Health check

> [!NOTE]
    > Skip this section if you're using Claude or OpenAI

```
 curl -v http://localhost:8000/health
```

It will show `200` status:
```
*   Trying 127.0.0.1:8000...
* Connected to localhost (127.0.0.1) port 8000 (#0)
> GET /health HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.81.0
> Accept: */*
> 
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< date: Fri, 25 Jul 2025 14:21:52 GMT
< server: uvicorn
< content-length: 0
< 
* Connection #0 to host localhost left intact
```

# Basic Usage for vLLM deployed Mistral

> [!Reference]
    > See notebooks in : [example notebooks](https://github.com/xinyu-dev/vrdc_text2sql/tree/main/examples)

> [!NOTE]
    > Skip this section if you're using Claude or OpenAI

## Create client

```python
import os
from openai import OpenAI
from loguru import logger
import sys
sys.path.append(os.path.abspath(os.path.join('..', 'model_evaluation')))

# The model path you used to start the vLLM server
MODEL_PATH = "/home/ubuntu/workspace/mistral-nemo-minitron-8b-instruct-healthcare-text2sql_vV2.8"

# vLLM server details from your running instance
IP = "localhost"
PORT = 8000
BASE_URL = f"http://{IP}:{PORT}/v1"

# Initialize the OpenAI client to connect to your local vLLM server
client = OpenAI(
    api_key="not-needed",  # The API key is not required for local server
    base_url=BASE_URL,
)
```

## Compose DDL statement

> [!NOTE]
    > The quality of DDL statement matters a lot to final accuracy! Make sure your DDL statement is properly annotated.

```python
# Simple DDL (Data Definition Language) for table schema
ddl = """\
CREATE TABLE patients (
    patient_id INT PRIMARY KEY,
    name VARCHAR(255),
    age INT,
    disease VARCHAR(255)
);
```

## Prepare chat

> [!tip]
    > If context window allow, embed DDL statement directly into the prompt

```python
# Instruction for the model
instruction = "Generate a SQLite query to answer the following question."

# The user's question
question = "How many patients are older than 50?"

# Format the prompt using the chat template from mistral-vllm.py
prompt_chat_template = [
    {
        "role": "system",
        "content": f"Based on DDL statements, instructions, and the current date, generate a SQL query in the following sqlite to answer the question:\n\nDDL statements:\n{ddl}\nInstructions:\n{instruction}",
    },
    {
        "role": "user",
        "content": f"{question}"
    }
]
```

## Run

```python
logger.info("Sending request to vLLM server...")

try:
    response = client.chat.completions.create(
        model=MODEL_PATH,
        messages=prompt_chat_template,
        temperature=0.0,
        max_tokens=512,  # Maximum length of the generated SQL query
        stop=["<extra_id_1>"] # Optional: stop sequence if your model uses one
    )

    # --- Print the response ---
    generated_sql = response.choices[0].message.content
    logger.success("\n✅ Server responded successfully!")
    logger.info("\nGenerated SQL Query:")
    logger.info(generated_sql)

except Exception as e:
    logger.error(f"\n❌ An error occurred: {e}")
```

# Inference with RAG

> [!Reference]
    > See notebooks in : [example notebooks](https://github.com/xinyu-dev/vrdc_text2sql/tree/main/examples)

## Architecture

> [!NOTE]
    > See [diagram](https://www.mermaidchart.com/app/projects/69dcd468-6052-4bde-97e1-ea268a4ab08c/diagrams/dd82ba4e-4616-4382-928b-104af646ba3b/version/v0.1/edit)

## Step 1: Confirm text2SQL LLM is running

1. If you're using Claude or OpenAi, skip this step
2. If you're using locally deployed vLLM, confirm that your server still running:
	```
	curl -v http://localhost:8000/health
	```

## Step 2: Configure text2SQL model

Configure the text2SQL client here:

```python
# vrdc_text2sql/model_inference/rag/mistral-text2sql-rag-vllm.py
# For finetuned Mistral model: 
async_client = AsyncOpenAI(
	api_key=os.getenv("NGC_API_KEY"),
	base_url=f"http://{args.ip}:{args.port}/v1",
)
```

Alternatively, for Claude:

```python
# vrdc_text2sql/model_inference/rag/mistral-text2sql-rag-vllm.py
# Alternatively, For claude: 
async_client = AsyncOpenAI(
     api_key=os.environ['BEDROCK_OPENAI_API_KEY'],
     base_url=os.environ['BEDROCK_OPENAI_BASE_URL']
)
```

The `chat.completion` is implemented here:
```python
# vrdc_text2sql/model_inference/rag/mistral-text2sql-rag-vllm.py
response = await async_client.chat.completions.create(
	model=checkpoint_path,
	messages=messages,
	temperature=0.0,
	top_p=0.95,
	max_completion_tokens=max_seq_length,
)
```

The `checkpoint_path` is passed through the CLI command line:

For locally deployed Mistral, this is path to the weights folder:
```bash
--checkpoint_path /root/workspace/mistral-nemo-minitron-8b-healthcare-text2sql_v1.0
```

For Claude on Bedrock, this is the model inference profile:
```bash
--checkpoint_path us.anthropic.claude-sonnet-4-20250514-v1:0
```

## Step 3: Configure Q&A embedding model

To configure the client for Q&A embedding model, find this:

```python
# vrdc_text2sql/model_inference/rag/rag.py

# example using playground NIM API endpiont
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NGC_API_KEY"),
)
```

Code to create the embedding is defined here:

```python
# vrdc_text2sql/model_inference/rag/rag.py

# this embeds the Q&A database (index)
def generate_database_embeddings(database, embedding_model_name, embedding_cache_file, input_type="passage"):
	...
	if sample.embedding is None:
		sample.embedding = client.embeddings.create(
			input=sample.question,
			model=embedding_model_name,
			encoding_format="float",
			extra_body={"input_type": input_type, "truncate": "NONE"},
		).data[0].embedding
		# print("finish to create embedding")
    ...
```

And here:

```python
# vrdc_text2sql/model_inference/rag/rag.py

# this embed the query so that we can use to retrieve the most relevant Q&A
def generate_embeddings(context, embedding_model_name, input_type="passage"):
    embedding_vector = client.embeddings.create(
        input=context,
        model=embedding_model_name,
        encoding_format="float",
        extra_body={"input_type": input_type, "truncate": "NONE"},
    ).data[0].embedding
```

The parameter `embedding_model_name` is specified through the CLI:

```bash
--embedding_model_name nvidia/nv-embedqa-mistral-7b-v2
```

Some embedding models (e..g `nv-embedqa-mistral-7b-v2`, `llama-3.2-nv-embedqa-1b-v2 ` ) requires the `extra_body` argument
```python
 extra_body={"input_type": input_type, "truncate": "NONE"}
```

> [!NOTE]
    > Some embedding model ask you to specify a `input_type` as either `passage` or `query`. The type of vector retrieval in the RAG here is not query/passage retrieval. Instead, it compares question to question, to retrieve similar Q&A pairs from the example dataset. Hence the input type should be both `passage` for embedding the dataset and embedding the query.

In `generate_database_embeddings` function, we pass `embedding_cache_file` as an argument. If this is a valid file path, then we use it as the pre-generated embedding. If is not valid file path, we will create a new embedding and cache it at this file path. You can set it with the CLI command:

```bash
--embedding_cache_file model_evaluation/dataset/train_eval/eicu/train_database_nv_embedcode_7b.pkl \
```

> [!NOTE]
    > Note that we recommend storing the index (e.g. embeddings of the Q&A pairs) to speed up retrieval. The embedding for the query should be generated on the fly.

## Step 4: Configure DDL embedding model

> [!NOTE]
    > Optional step. Only required if you plan to add DDL RAG

Configure the client here:
```python
# vrdc_text2sql/model_evaluation/utils/experimental.py
class FAISSRetriever:
    def __init__(self, api_key, endpoint, api_version=None, model="text-embedding-3-large"):
        """
        Initialize the retriever with NVIDIA API client
        
        Args:
            api_key: NVIDIA API key (if None, will look for NGC_API_KEY in env)
            model: Embedding model name
        """

        # == Create a client instance for embedding ==
        # use Azure OpenAI for embeding
        # self.client = AzureOpenAI(
        #     api_key=api_key,
        #     api_version=api_version,
        #     azure_endpoint=endpoint
        # )

        # use NIM for embeding
        self.client = OpenAI(
            api_key = api_key,
            base_url = endpoint
        )
        # == End of creating a client instance for embedding ==
```

The `api_key` and `endpoint` are specified here:

```python
# vrdc_text2sql/model_inference/rag/mistral-text2sql-rag-vllm.py

# example using playground NIM API endpiont
ddl_retreiver = FAISSRetriever(
	api_key=os.getenv("NGC_API_KEY"),
	endpoint = "https://integrate.api.nvidia.com/v1",
	model = "nvdev/nvidia/llama-3.2-nv-embedqa-1b-v2",  # remove the "nvdev/" if using public playground 
)
```

Code to generate embedding is specified below. This same block of code is used to create both the index (chunked DDL statement) and the question.

```python
# vrdc_text2sql/model_evaluation/utils/experimental.py
def generate_embedding(self, text):
	"""Generate embedding for a single text"""
	response = self.client.embeddings.create(
		input=text,
		model=self.model, 
		encoding_format="float",
		extra_body={"input_type": "passage", "truncate": "NONE"}
	)
	return response.data[0].embedding
```

> [!NOTE]
    > Some embedding model ask you to specify a `input_type` as either `passage` or `query`. The type of vector retrieval in the RAG here is not query/passage retrieval. Instead, it compares question to question, to retrieve similar Q&A pairs from the example dataset. Hence the input type should be both `passage` for embedding the dataset and embedding the query.

> [!NOTE]
    > Similar to the Q&A embeddings, we recommend storing the index (e.g. embeddings of DDL chunks) to speed up retrieval.

Change path to this DDL embedding file is specified here:

```python
# vrdc_text2sql/model_inference/rag/mistral-text2sql-rag-vllm.py
ddl_embedding_cache_file = "/root/workspace/vrdc_text2sql/model_evaluation/dataset/train_eval/eicu/ddl_database_llama-3.2-nv-embedqa-1b-v1.pkl"
```

If this is a valid file path, then we use it as the pre-generated embedding. If is not valid file path, we will create a new embedding and cache it at this file path.

## Step 5: Configure DDL chunking method

> [!warning]
    > Inspect the chunking mechanism carefully. Depend on how your DDL script is written, you might want to change it accordingly. The key is to preserve each DDL statement (each table schema) **intactly** during chunking.

```python fold
# vrdc_text2sql/model_evaluation/utils/experimental.py

def split_sql_blocks(file_path):
    """
    file_path: path to the `eicu_instruct_benchmark_rag.sql` file
    Read an SQL file and split it into blocks of code.
    Each block contains a DROP TABLE and CREATE TABLE statement for one table.
    """
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split by "DROP TABLE IF EXISTS" statements
    # This pattern looks for DROP TABLE at the start of a line
    pattern = r'^DROP TABLE IF EXISTS'
    
    # Find all positions where DROP TABLE statements start
    lines = content.split('\n')
    block_starts = []
    
    for i, line in enumerate(lines):
        if re.match(pattern, line.strip()):
            block_starts.append(i)
    
    # Add the end of file as the last position
    block_starts.append(len(lines))
    
    # Extract blocks
    blocks = []
    for i in range(len(block_starts) - 1):
        start_line = block_starts[i]
        end_line = block_starts[i + 1]
        
        # Join lines for this block
        block_lines = lines[start_line:end_line]
        
        # Remove empty lines at the end of the block
        while block_lines and block_lines[-1].strip() == '':
            block_lines.pop()
        
        if block_lines:
            block = '\n'.join(block_lines)
            blocks.append(block)
    
    return blocks
```

If you modify the code above, just make sure you return the data type as `List[str]`. For example:

```python
# split_sql_blocks should contain chunks of DDL, where each chunk has an intact table schema
[
	"Create TABLE A......;", 
	"Create TABLE B......;"
]
```

## Step 6: Run prediction

> [!NOTE]
    > We provide example evaluation file `model_inference/rag/mistral-text2sql-rag-vllm.py`.

### Example of locally deployed finetuned Mistral with QA RAG (no DDL RAG)

```bash
# create the output directory where you want to store the results
mkdir -p model_predictions/eICU/rag/mistral_finetuned

# tweak the command as needed for your case
python model_inference/rag/mistral-text2sql-rag-vllm.py \
	--task_name ehrsql_eicu \
	--ip localhost \
	--port 8000 \
	--checkpoint_path /root/workspace/mistral-nemo-minitron-8b-healthcare-text2sql_v1.0 \
	--max_seq_length 4096 \
	--batch_size 4 \
	--save_dir model_predictions/eICU/rag/mistral_finetuned \
	--train_file model_evaluation/dataset/train_eval/eicu/train_val.csv \
	--embedding_model_name nvidia/nv-embedqa-mistral-7b-v2 \
	--embedding_cache_file model_evaluation/dataset/train_eval/eicu/train_database_nv_embedcode_7b.pkl \
	--top_k 3 \
	--dataset_path model_evaluation/dataset/test/test_ehrsql_eicu_data_benchmark_rag.json \
	--metadata_path /root/workspace/vrdc_text2sql/model_evaluation/dataset/metadata/eicu_instruct_benchmark_rag.sql \
	--format sqlite
```

Note the missing `--experimental` tag. This will skip DDL RAG. Instead, it will drop the entire DDL schema of all tables into the prompt.

### Example of locally deployed finetuned Mistral with DDL RAG + QA RAG

> [!warning]
    > Configurations for DDL RAG is located in [[#Step 4 Configure DDL embedding model]] and [[#Step 5 Configure DDL chunking method]]. These configurations need to be changed in the scripts directly, and they are not directly passed through the CLI command

```bash
# create the output directory where you want to store the results
mkdir -p model_predictions/eICU/rag/mistral_finetuned_nv-embedqa_ddl5_qa6

# tweak the command as needed for your case
python model_inference/rag/mistral-text2sql-rag-vllm.py \
	--task_name ehrsql_eicu \
	--ip localhost \
	--port 8000 \
	--checkpoint_path /root/workspace/mistral-nemo-minitron-8b-healthcare-text2sql_v1.0 \
	--max_seq_length 4096 \
	--batch_size 4 \
	--save_dir model_predictions/eICU/rag/mistral_finetuned_nv-embedqa_ddl5_qa6 \
	--train_file model_evaluation/dataset/train_eval/eicu/train_val.csv \
	--embedding_model_name nvidia/nv-embedqa-mistral-7b-v2 \
	--embedding_cache_file model_evaluation/dataset/train_eval/eicu/train_database_nv_embedcode_7b.pkl \
	--top_k 6 \
	--dataset_path model_evaluation/dataset/test/test_ehrsql_eicu_data_benchmark_rag.json \
	--metadata_path /root/workspace/vrdc_text2sql/model_evaluation/dataset/metadata/eicu_instruct_benchmark_rag.sql \
	--format sqlite \
	--experimental
```

### Example of running Claude 4 with DDL RAG + QA RAG

> [!warning]
    > Configurations for DDL RAG is located in [[#Step 4 Configure DDL embedding model]] and [[#Step 5 Configure DDL chunking method]]. These configurations need to be changed in the scripts directly, and they are not directly passed through the CLI command

```bash
# create the output directory where you want to store the results
mkdir -p model_predictions/eICU/rag/claude_sonnet_4_no_thinking_nv-embedqa_ddl5_qa6

# tweak the command as needed for your case
python model_inference/rag/mistral-text2sql-rag-vllm.py \
	--task_name ehrsql_eicu \
	--ip localhost \
	--port 8000 \
	--checkpoint_path us.anthropic.claude-sonnet-4-20250514-v1:0 \
	--max_seq_length 4096 \
	--batch_size 4 \
	--save_dir model_predictions/eICU/rag/claude_sonnet_4_no_thinking_nv-embedqa_ddl5_qa6 \
	--train_file model_evaluation/dataset/train_eval/eicu/train_val.csv \
	--embedding_model_name nvidia/nv-embedqa-mistral-7b-v2 \
	--embedding_cache_file model_evaluation/dataset/train_eval/eicu/train_database_nv_embedcode_7b.pkl \
	--top_k 6 \
	--dataset_path model_evaluation/dataset/test/test_ehrsql_eicu_data_benchmark_rag.json \
	--metadata_path /root/workspace/vrdc_text2sql/model_evaluation/dataset/metadata/eicu_instruct_benchmark_rag.sql \
	--format sqlite \
	--experimental
```

### Example of running Claude 3.7 with DDL RAG + QA RAG

> [!warning]
    > Configurations for DDL RAG is located in [[#Step 4 Configure DDL embedding model]] and [[#Step 5 Configure DDL chunking method]]. These configurations need to be changed in the scripts directly, and they are not directly passed through the CLI command

```bash
# create the output directory where you want to store the results
mkdir -p model_predictions/eICU/rag/claude_sonnet_3_7_no_thinking_nv-embedqa_ddl5_qa6

# tweak the command as needed for your case
python model_inference/rag/mistral-text2sql-rag-vllm.py \
	--task_name ehrsql_eicu \
	--ip localhost \
	--port 8000 \
	--checkpoint_path us.anthropic.claude-3-7-sonnet-20250219-v1:0 \
	--max_seq_length 4096 \
	--batch_size 4 \
	--save_dir model_predictions/eICU/rag/claude_sonnet_3_7_no_thinking_nv-embedqa_ddl5_qa6 \
	--train_file model_evaluation/dataset/train_eval/eicu/train_val.csv \
	--embedding_model_name nvidia/nv-embedqa-mistral-7b-v2 \
	--embedding_cache_file model_evaluation/dataset/train_eval/eicu/train_database_nv_embedcode_7b.pkl \
	--top_k 6 \
	--dataset_path model_evaluation/dataset/test/test_ehrsql_eicu_data_benchmark_rag.json \
	--metadata_path /root/workspace/vrdc_text2sql/model_evaluation/dataset/metadata/eicu_instruct_benchmark_rag.sql \
	--format sqlite \
	--experimental
```

### Explanation of Argument

**Key Parameters:**

- `--task_name`: Dataset name (e.g., ehrsql_eicu, mimicsql)
- `--ip` and `--port`: vLLM server connection details, for the text2SQL model. If you're using Claude, these arguments don't apply. Just enter whatever number.
- `--checkpoint_path`: Path to your fine-tuned model. For vLLM deployed model, specify the path to the weights folder. For Claude deployed on bedrock, specify the inference profile name. For OpenAI model, specify the model name.
- `--train_file`: CSV file with training examples for RAG retrieval
- `--embedding_model_name`: Embedding model for vector search for the **Q&A RAG retrieval**. For example: `nvidia/nv-embedqa-mistral-7b-v2` or `text-embedding-3-large`. The value will be used as the `model` parameter for the LLM server
- `--embedding_cache_file`: Cache file for pre-computed embedding pickle file for **Q&A RAG retrieval**. If file does not exist, new embedding file will be created.
- `--top_k`: Number of similar examples to retrieve (default: 3) for **Q&A RAG retrieval**.
- `--dataset_path`: Test dataset JSON file
- `--metadata_path`: Database schema/metadata file of the test dataset.
- `--experimental`: Boolean. When this tag is included, it activates the experimental feature that perform **chunking on DDL then RAG retrieval** of the most relevant schema chunks. Without `--experimental` tag, the pipeline will only perform RAG on Q&A pairs. If activated, you must use a model that has large input window, otherwise individual DDL blocks might not be split intact. This embedding model does not need to be the same as the embedding model used for **Q&A RAG retrieval**. For example, we use `nvidia/nv-embedqa-mistral-7b-v2` for **Q&A RAG retrieval**, and OpenAI `text-embedding-3-large` for **DDL RAG retrieval**

**Outputs:**

- Model predictions in JSONL format
- System performance metrics (memory, latency, throughput)
- FAISS index cache for faster subsequent runs

**Performance Optimization:**

- Adjust `--batch_size` based on GPU memory
- Use `--top_k` to control retrieval context size
- Monitor memory usage and adjust `--max-num-seqs` accordingly

## Step 7. Evaluate

### Explanation of Metrics

They can be grouped into two main categories: **Answerability Metrics** (`_ans`) and **Execution Metrics** (`_exec`), plus overall accuracy.

#### 1. Answerability Metrics (`_ans`)

These metrics **ignore the correctness of the SQL query**. They only care about whether the model correctly identified if a question was *answerable* or *unanswerable*. **A question is considered answerable if its ground-truth query can be executed without error.**

- `precision_ans`: **"Of all the questions the model *claimed* were answerable, what percentage actually were?"**
    - This measures how much you can trust the model when it generates a query. A high score means it doesn't often generate queries for questions that are impossible to answer.
- `recall_ans`: **"Of all the questions that *truly were* answerable, what percentage did the model *attempt* to answer?"**
    - This measures how comprehensive the model is. A high score means the model doesn't frequently give up on questions it should have been able to answer
- `f1_ans`: This is the **F1-score for answerability**.
    - It's the harmonic mean of `precision_ans` and `recall_ans`, providing a single score that balances the two. It gives a good overall sense of how well the model distinguishes answerable from unanswerable questions.

#### 2. Execution Metrics (`_exec`)

These are the most important metrics. They evaluate the model's performance on the questions that are **supposed to be answerable**. They check if the *result* of the predicted SQL query matches the *result* of the ground-truth SQL query.

- `precision_exec`: **"Of all the answerable questions the model *attempted* to answer, what percentage of its SQL queries were correct?"**
    - This is a direct measure of the model's correctness. When the model generates a query, how often is that query right?
- `recall_exec`: **"Of all the answerable questions in the dataset, what percentage did the model answer correctly?"**
    - This measures how much of the total task the model accomplished successfully. It penalizes the model for both failing to answer a question it should have and for getting the answer wrong.
- `f1_exec`: This is the **F1-score for execution**.
    - It's the harmonic mean of `precision_exec` and `recall_exec`. This is often considered the primary metric for evaluating the overall quality of the text-to-SQL model.

#### 3. Overall Accuracy (`acc`)

`acc`: **"What is the overall percentage of questions that the model got right?"**
- This is the simplest metric. It considers a prediction correct if the result matches the ground-truth result, or if both correctly failed (i.e., for unanswerable questions). It provides a good, high-level summary of performance across all questions, both answerable and unanswerable.

### Run evaluation

> [!NOTE]
    > See `vrdc_text2sql/examples/6.RAG.ipynb`

1. First, create a directory to store the evaluation results:
	```python
	import os, sys
	sys.path.append(os.path.abspath(os.path.join('..', 'model_evaluation')))
	from utils import preprare_directory
	from dotenv import load_dotenv
	load_dotenv()
	
	# create output directory for evaluation results, relative to the path of model_evaluation directory
	# note that the evaluate results need a clean new folder, because it will overwrite any existing files in the folder
	pred_directory = f"/root/workspace/vrdc_text2sql/model_predictions/eICU/rag/claude_sonnet_4_no_thinking_nv-embedqa_ddl5_qa6"  
	eval_directory = os.path.join(pred_directory, "evaluation")
	preprare_directory(eval_directory, exist_ok=False)
	
	# path to the prediction results
	pred_file = f"{pred_directory}/test_rag_vllm_ehrsql_eicu_result_mis_embedd.jsonl"
	print("Using predictions from: ", pred_file)
	
	# path to the eICU database
	db_path = "/root/workspace/vrdc_text2sql/model_evaluation/databases/eicu.sqlite"
	```
2. Run the evaluation:
	```python
	# run evaluation
	!python ../model_evaluation/ehrsql_eval.py \
	    --pred_file {pred_file} \
	    --db_path {db_path} \
	    --num_workers -1 \
	    --timeout 60 \
	    --out_file {eval_directory} \
	    --ndigits 2
	```
	- `--pred_file`: Path to model predictions (JSON/JSONL)
	- `--db_path`: Path to SQLite database file
	- `--num_workers`: Parallel workers (default: -1 for all CPUs)
	- `--timeout`: Query execution timeout (seconds)
	- `--out_file`: Output directory for results and metrics
	- `--ndigits`: Number of digits to round scores (default: 2)
	- Outputs:
		- Per-sample evaluation results (JSONL)
		- Overall metrics (precision, recall, F1, accuracy) in JSON
		- Error logs for failed queries
3. We can view the metrics file here:
	```python
	import json
	
	# file path to the evaluation result file. 
	fp = f"{pred_directory}/evaluation/test_rag_vllm_ehrsql_eicu_result_mis_embedd_metrics.json"
	print("Reading from file: ", fp)
	
	with open(fp, "r") as f:
	    metrics = json.load(f)
	
	print(json.dumps(metrics, indent=4))
	```
4. Output looks like this:
	```json
	{
	    "precision_ans": 100.0,
	    "recall_ans": 100.0,
	    "f1_ans": 100.0,
	    "precision_exec": 93.42,
	    "recall_exec": 93.42,
	    "f1_exec": 93.42,
	    "acc": 93.42
	}
	```
5. 

# Caveats

1. DDL chunking on eICU schema is simply done by looking at the `DROP TALBE IF EXISTS` statement. For other type of schema, you might need to tweak it so that it chunks individual blocks of SQL in an intact way. This is very important. To implement your own DDL chunking logic, modify the `split_sql_blocks` function:
	```python
	# vrdc_text2sql/model_evaluation/utils/experimental.py
	def split_sql_blocks(file_path):
	    """
	    file_path: path to the `eicu_instruct_benchmark_rag.sql` file
	    Read an SQL file and split it into blocks of code.
	    Each block contains a DROP TABLE and CREATE TABLE statement for one table.
	    """
	    ...
	```
2. The embedding model used for DDL chunking must have a large context window, otherwise it might run out of context. `nvidia/nv-embedqa-mistral-7b-v2` has a window of **512** token, which is too small. OpenAI's `text-embedding-3-large` has a **8191** token.
3. 
