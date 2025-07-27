export PYTHONPATH=$PYTHONPATH:$(pwd)



# #----------------------------------------------------------------------------------------------------
# # Base models: mistral-nemo-minitron-8b-instruct
# #----------------------------------------------------------------------------------------------------
# pred_file="model_predictions/model_base/mistral-nemo-minitron-8b-instruct/mistral-nemo-minitron-8b-instruct_vinmec_results.json"
# echo "Evaluating $pred_file"
# python psql_sql_eval.py \
#     --config configs/sql_eval.yaml \
#     --pred_file "$pred_file" \
#     --out_file results \
#     --input_template chat
# # all: exactly: 0.3796020679931067, subset: 0.3819520601597995
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------


# #----------------------------------------------------------------------------------------------------
# # Base models: llama31-nemotron-nano-8b-v1 (no thinking)
# #----------------------------------------------------------------------------------------------------
# pred_file="model_predictions/model_base/llama31-nemotron-nano-8b-v1/llama31-nemotron-nano-8b-v1_vinmec_results 1.json"
# echo "Evaluating $pred_file"
# python psql_sql_eval.py \
#     --config configs/sql_eval.yaml \
#     --pred_file "$pred_file" \
#     --out_file results \
#     --input_template chat
# # all: exactly: 0.031271430530791386, subset: 0.031271430530791386
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------



# #----------------------------------------------------------------------------------------------------
# # Base models: llama31-nemotron-nano-8b-v1 (thinking)
# #----------------------------------------------------------------------------------------------------
# pred_file="model_predictions/model_base/llama31-nemotron-nano-8b-v1/llama31-nemotron-nano-8b-v1_vinmec_results_thinking 1.json"
# echo "Evaluating $pred_file"
# python psql_sql_eval.py \
#     --config configs/sql_eval.yaml \
#     --pred_file "$pred_file" \
#     --out_file results \
#     --input_template chat
# # all: exactly: 0.5174274464051335, subset: 0.5254484468426426
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------


# #----------------------------------------------------------------------------------------------------
# # Finetuned models: mistral-nemo-minitron-8b-instruct (single task + training benchmark data)
# #----------------------------------------------------------------------------------------------------
# pred_file="model_predictions/test_mistral-text2sql-with-benchmark-vinmec_result.jsonl"
# echo "Evaluating $pred_file"
# python psql_sql_eval.py \
#     --config configs/sql_eval.yaml \
#     --pred_file "$pred_file" \
#     --out_file results \
#     --input_template string
# # all: exactly: 0.5174274464051335, subset: 0.5254484468426426
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------


# #----------------------------------------------------------------------------------------------------
# # Finetuned models: mistral-nemo-minitron-8b-instruct (single task)
# #----------------------------------------------------------------------------------------------------
# pred_file="model_predictions/test_mistral-text2sql-vinmec_result.jsonl"
# echo "Evaluating $pred_file"
# python psql_sql_eval.py \
#     --config configs/sql_eval.yaml \
#     --pred_file "$pred_file" \
#     --out_file results \
#     --input_template string
# # all: exactly: 0.5174274464051335, subset: 0.5254484468426426
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------


# #----------------------------------------------------------------------------------------------------
# # Finetuned models: mistral-nemo-minitron-8b-instruct (single task)
# #----------------------------------------------------------------------------------------------------
# pred_file="model_predictions/text_mistral-text2sql-with-benchmark-vinmec_result.jsonl"
# echo "Evaluating $pred_file"
# python psql_sql_eval.py \
#     --config configs/sql_eval.yaml \
#     --pred_file "$pred_file" \
#     --out_file results \
#     --input_template string
# # all: exactly: 0.5174274464051335, subset: 0.5254484468426426
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------


# #----------------------------------------------------------------------------------------------------
# # Base models: llama3-sqlcoder-8b
# #----------------------------------------------------------------------------------------------------
# pred_file="model_predictions/model_base/llama3-sqlcoder-8b/llama3-sqlcoder-8b_vinmec_results.json"
# echo "Evaluating $pred_file"
# python psql_sql_eval.py \
#     --config configs/sql_eval.yaml \
#     --pred_file "$pred_file" \
#     --out_file results \
#     --input_template string
# # all: exactly: 0.29219570001692907, subset: 0.29219570001692907
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------



# #----------------------------------------------------------------------------------------------------
# # Base models: mistral-nemo-minitron-8b-instruct
# #----------------------------------------------------------------------------------------------------
# pred_file="model_predictions/model_base/mistral-nemo-minitron-8b-instruct/mistral-nemo-minitron-8b-instruct_vinmec_results.json"
# echo "Evaluating $pred_file"
# python psql_sql_eval.py \
#     --config configs/sql_eval.yaml \
#     --pred_file "$pred_file" \
#     --out_file results \
#     --input_template chat
# # all: exactly: 0.3796020679931067, subset: 0.3819520601597995
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------



# #----------------------------------------------------------------------------------------------------
# # Base models: Gwen/QwQ-32B
# #----------------------------------------------------------------------------------------------------
# pred_file="model_predictions/model_base/QwQ-32B/qwq-32b_vinmec_results.json"
# echo "Evaluating $pred_file"
# python psql_sql_eval.py \
#     --config configs/sql_eval.yaml \
#     --pred_file "$pred_file" \
#     --out_file results \
#     --input_template chat
# # all: exactly: 0.5174274464051335, subset: 0.5254484468426426
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------



# #----------------------------------------------------------------------------------------------------
# # Base models: llama31-nemotron-nano-8b-v1 (no thinking)
# #----------------------------------------------------------------------------------------------------
# pred_file="model_predictions/model_base/llama31-nemotron-nano-8b-v1/llama31-nemotron-nano-8b-v1_vinmec_results 1.json"
# echo "Evaluating $pred_file"
# python psql_sql_eval.py \
#     --config configs/sql_eval.yaml \
#     --pred_file "$pred_file" \
#     --out_file results \
#     --input_template chat
# # all: exactly: 0.031271430530791386, subset: 0.031271430530791386
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------



# #----------------------------------------------------------------------------------------------------
# # Base models: llama31-nemotron-nano-8b-v1 (thinking)
# #----------------------------------------------------------------------------------------------------
# pred_file="model_predictions/model_base/llama31-nemotron-nano-8b-v1/llama31-nemotron-nano-8b-v1_vinmec_results_thinking 1.json"
# echo "Evaluating $pred_file"
# python psql_sql_eval.py \
#     --config configs/sql_eval.yaml \
#     --pred_file "$pred_file" \
#     --out_file results \
#     --input_template chat
# # all: exactly: 0.5174274464051335, subset: 0.5254484468426426
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------


# #----------------------------------------------------------------------------------------------------
# # Finetuned models: mistral-nemo-minitron-8b-instruct (single task + training benchmark data)
# #----------------------------------------------------------------------------------------------------
# pred_file="model_predictions/test_mistral-text2sql-with-benchmark-vinmec_result.jsonl"
# echo "Evaluating $pred_file"
# python psql_sql_eval.py \
#     --config configs/sql_eval.yaml \
#     --pred_file "$pred_file" \
#     --out_file results \
#     --input_template string
# # all: exactly: 0.5174274464051335, subset: 0.5254484468426426
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------


# #----------------------------------------------------------------------------------------------------
# # Finetuned models: mistral-nemo-minitron-8b-instruct (single task)
# #----------------------------------------------------------------------------------------------------
# pred_file="model_predictions/test_mistral-text2sql-vinmec_result.jsonl"
# echo "Evaluating $pred_file"
# python psql_sql_eval.py \
#     --config configs/sql_eval.yaml \
#     --pred_file "$pred_file" \
#     --out_file results \
#     --input_template string
# # all: exactly: 0.5174274464051335, subset: 0.5254484468426426
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------
# Base model: Qwen2.5-Coder-7B-Instruct (text2SQL)
#----------------------------------------------------------------------------------------------------
pred_file="model_predictions/model_base/Qwen2.5-Coder-7B-Instruct/Qwen2.5-Coder-7B-Instruct_vinmec_result.jsonl"
echo "Evaluating $pred_file"
python psql_sql_eval.py \
    --config configs/sql_eval.yaml \
    --pred_file "$pred_file" \
    --out_file outputs/model_base/Qwen2.5-Coder-7B-Instruct \
    --input_template string
# all: exactly: 0.5174274464051335, subset: 0.5254484468426426
echo "Done\n" 
#----------------------------------------------------------------------------------------------------