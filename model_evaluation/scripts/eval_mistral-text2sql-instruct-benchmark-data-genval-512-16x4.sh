#----------------------------------------------------------------------------------------------------
# Finetuned models: mistral-nemo-minitron-8b-instruct (single task - Remove EHR-2024-MIMIC-IV dataset- tokens to generate: 512)
#----------------------------------------------------------------------------------------------------
export PYTHONPATH=.

# pred_file="model_predictions/hvnguyen/mistral-text2sql-v24/test_mistral-text2sql-instruct-benchmark-data-genval-512-16x4-ehrsql-mimicsql_result.jsonl"
# echo "Evaluating $pred_file"
# python mimicsql_eval.py \
#     --pred_file "$pred_file" \
#     --db_path databases/mimic_all.db \
#     --num_workers -1 \
#     --timeout 60 \
#     --out_file outputs/mistral-text2sql-instruct-benchmark-data-genval-512-16x4 \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------


# pred_file="model_predictions/hvnguyen/mistral-text2sql-v24/test_mistral-text2sql-instruct-benchmark-data-genval-512-16x4-ehrsql-eicu_result.jsonl"
# echo "Evaluating $pred_file"
# python ehrsql_eval.py \
#     --pred_file "$pred_file" \
#     --db_path databases/eicu.sqlite \
#     --num_workers 1 \
#     --timeout 60 \
#     --out_file outputs/mistral-text2sql-instruct-benchmark-data-genval-512-16x4 \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------


# pred_file="model_predictions/hvnguyen/mistral-text2sql-v24/test_mistral-text2sql-instruct-benchmark-data-genval-512-16x4-ehrsql-mimic-iii_result.jsonl"
# echo "Evaluating $pred_file"
# python ehrsql_eval.py \
#     --pred_file "$pred_file" \
#     --db_path databases/mimic_iii.sqlite \
#     --num_workers 1 \
#     --timeout 60 \
#     --out_file outputs/mistral-text2sql-instruct-benchmark-data-genval-512-16x4 \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------


# pred_file="model_predictions/hvnguyen/mistral-text2sql-v24/test_mistral-text2sql-instruct-benchmark-data-genval-512-16x4-vinmec_result.jsonl"
# echo "Evaluating $pred_file"
# python psql_sql_eval.py \
#     --config configs/sql_eval.yaml \
#     --pred_file "$pred_file" \
#     --out_file outputs/mistral-text2sql-instruct-benchmark-data-genval-512-16x4 \
#     --input_template string
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------
