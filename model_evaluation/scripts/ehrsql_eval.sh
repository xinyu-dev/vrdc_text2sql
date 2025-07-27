export PYTHONPATH=.

# #----------------------------------------------------------------------------------------------------
# # Finetuned models: mistral-nemo-minitron-8b-instruct (single task)
# #----------------------------------------------------------------------------------------------------
# pred_file="model_predictions/test_mistral-text2sql-ehrsql-eicu_result.jsonl"
# echo "Evaluating $pred_file"
# python ehrsql_eval.py \
#     --pred_file $pred_file \
#     --db_path databases/eicu.sqlite \
#     --num_workers -1 \
#     --timeout 60 \
#     --out_file outputs \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------

# pred_file="model_predictions/test_mistral-text2sql-ehrsql-mimic_result.jsonl"
# echo "Evaluating $pred_file"
# python ehrsql_eval.py \
#     --pred_file $pred_file \
#     --db_path databases/mimic_iii.sqlite \
#     --num_workers -1 \
#     --timeout 60 \
#     --out_file outputs \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------

# pred_file="model_predictions/test_mistral-text2sql-mimicsql_result.jsonl"
# echo "Evaluating $pred_file"
# python ehrsql_eval.py \
#     --pred_file $pred_file \
#     --db_path databases/mimic_all.db \
#     --num_workers -1 \
#     --timeout 60 \
#     --out_file outputs \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------



# #----------------------------------------------------------------------------------------------------
# # Finetuned models: mistral-nemo-minitron-8b-instruct (multi-task)
# #----------------------------------------------------------------------------------------------------
# pred_file="model_predictions/test_mistral-3-tasks-ehrsql-eicu_result.jsonl"
# echo "Evaluating $pred_file"
# python ehrsql_eval.py \
#     --pred_file $pred_file \
#     --db_path databases/eicu.sqlite \
#     --num_workers -1 \
#     --timeout 60 \
#     --out_file outputs \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------

# pred_file="model_predictions/test_mistral-3-tasks-ehrsql-mimic_result.jsonl"
# echo "Evaluating $pred_file"
# python ehrsql_eval.py \
#     --pred_file $pred_file \
#     --db_path databases/mimic_iii.sqlite \
#     --num_workers -1 \
#     --timeout 60 \
#     --out_file outputs \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------

# pred_file="model_predictions/test_mistral-3-tasks-mimicsql_result.jsonl"
# echo "Evaluating $pred_file"
# python ehrsql_eval.py \
#     --pred_file $pred_file \
#     --db_path databases/mimic_all.db \
#     --num_workers -1 \
#     --timeout 60 \
#     --out_file outputs \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------

# pred_file="model_predictions/mistral-text2sql_ehrsql_eicu_results_new_prompt.json"
# echo "Evaluating $pred_file"
# python ehrsql_eval.py \
#     --pred_file $pred_file \
#     --db_path databases/eicu.sqlite \
#     --num_workers -1 \
#     --timeout 60 \
#     --out_file outputs \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------
# #----------------------------------------------------------------------------------------------------



# #----------------------------------------------------------------------------------------------------
# # Finetuned models: mistral-nemo-minitron-8b-instruct (single task + training benchmark data)
# #----------------------------------------------------------------------------------------------------
# pred_file="model_predictions/test_mistral-text2sql-with-benchmark-ehrsql-eicu_result.jsonl"
# echo "Evaluating $pred_file"
# python ehrsql_eval.py \
#     --pred_file $pred_file \
#     --db_path databases/eicu.sqlite \
#     --num_workers -1 \
#     --timeout 60 \
#     --out_file outputs \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------

# pred_file="model_predictions/test_mistral-text2sql-with-benchmark-ehrsql-mimic_result.jsonl"
# echo "Evaluating $pred_file"
# python ehrsql_eval.py \
#     --pred_file $pred_file \
#     --db_path databases/mimic_iii.sqlite \
#     --num_workers -1 \
#     --timeout 60 \
#     --out_file outputs \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------

# pred_file="model_predictions/test_mistral-text2sql-with-benchmark-mimicsql_result.jsonl"
# echo "Evaluating $pred_file"
# python ehrsql_eval.py \
#     --pred_file $pred_file \
#     --db_path databases/mimic_all.db \
#     --num_workers -1 \
#     --timeout 60 \
#     --out_file outputs \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------
# #----------------------------------------------------------------------------------------------------




# #----------------------------------------------------------------------------------------------------
# # Base models: llama31-nemotron-nano-8b-v1 (thinking)
# #----------------------------------------------------------------------------------------------------
# pred_file="model_predictions/model_base/llama31-nemotron-nano-8b-v1/llama31-nemotron-nano-8b-v1_ehrsql_eicu_results_thinking.json"
# echo "Evaluating $pred_file"
# python ehrsql_eval.py \
#     --pred_file "$pred_file" \
#     --db_path databases/eicu.sqlite \
#     --num_workers -1 \
#     --timeout 60 \
#     --out_file outputs \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------

# pred_file="model_predictions/model_base/llama31-nemotron-nano-8b-v1/llama31-nemotron-nano-8b-v1_ehrsql_mimiciii_results_thinking.json"
# echo "Evaluating $pred_file"
# python ehrsql_eval.py \
#     --pred_file "$pred_file" \
#     --db_path databases/mimic_iii.sqlite \
#     --num_workers -1 \
#     --timeout 60 \
#     --out_file outputs \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------

# pred_file="model_predictions/model_base/llama31-nemotron-nano-8b-v1/llama31-nemotron-nano-8b-v1_mimicsql_results_thinking.json"
# echo "Evaluating $pred_file"
# python ehrsql_eval.py \
#     --pred_file "$pred_file" \
#     --db_path databases/mimic_all.db \
#     --num_workers -1 \
#     --timeout 60 \
#     --out_file outputs \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------
# #----------------------------------------------------------------------------------------------------



# #----------------------------------------------------------------------------------------------------
# # Base models: llama31-nemotron-nano-8b-v1 (no thinking)
# #----------------------------------------------------------------------------------------------------
# pred_file="model_predictions/model_base/llama31-nemotron-nano-8b-v1/llama31-nemotron-nano-8b-v1_ehrsql_eicu_results.json"
# echo "Evaluating $pred_file"
# python ehrsql_eval.py \
#     --pred_file "$pred_file" \
#     --db_path databases/eicu.sqlite \
#     --num_workers -1 \
#     --timeout 60 \
#     --out_file outputs \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------

# pred_file="model_evaluation/model_predictions/model_base/llama31-nemotron-nano-8b-v1/llama31-nemotron-nano-8b-v1_ehrsql_mimic_results.json"
# echo "Evaluating $pred_file"
# python ehrsql_eval.py \
#     --pred_file "$pred_file" \
#     --db_path databases/mimic_iii.sqlite \
#     --num_workers -1 \
#     --timeout 60 \
#     --out_file outputs \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------

# pred_file="model_predictions/model_base/llama31-nemotron-nano-8b-v1/llama31-nemotron-nano-8b-v1_mimicsql_results_no_thinking.json"
# echo "Evaluating $pred_file"
# python ehrsql_eval.py \
#     --pred_file "$pred_file" \
#     --db_path databases/mimic_all.db \
#     --num_workers -1 \
#     --timeout 60 \
#     --out_file outputs \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------
# #----------------------------------------------------------------------------------------------------


# pred_file="model_predictions/test_mistral-lora-text2sql-instruct-benchmark-ehrsql-mimic_result.jsonl"
# echo "Evaluating $pred_file"
# python ehrsql_eval.py \
#     --pred_file "$pred_file" \
#     --db_path databases/mimic_iii.sqlite \
#     --num_workers -1 \
#     --timeout 60 \
#     --out_file outputs \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------




# pred_file="model_predictions/test_mistral-text2sql-instruct-benchmark-data-genval-128-16x4-ehrsql-mimicsql_result.jsonl"
# echo "Evaluating $pred_file"
# python ehrsql_eval.py \
#     --pred_file "$pred_file" \
#     --db_path databases/mimic_all.db \
#     --num_workers 1 \
#     --timeout 60 \
#     --out_file outputs \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------



pred_file="/Users/hvnguyen/projects/model-evaluation/test_mistral-text2sql-instruct-benchmark-data-genval-512-16x4-ehrsql-eicu_result.jsonl"
echo "Evaluating $pred_file"
python ehrsql_eval.py \
    --pred_file "$pred_file" \
    --db_path databases/eicu.sqlite \
    --num_workers 1 \
    --timeout 60 \
    --out_file outputs \
    --ndigits 2 
echo "Done\n" 
#----------------------------------------------------------------------------------------------------