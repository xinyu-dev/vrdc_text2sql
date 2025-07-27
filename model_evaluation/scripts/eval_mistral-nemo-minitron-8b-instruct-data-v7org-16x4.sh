#----------------------------------------------------------------------------------------------------
# models: Mistral-Nemo-Minitron-8B-Instruct, data version: v7org, resource: gpu 16x4 
#----------------------------------------------------------------------------------------------------
export PYTHONPATH=.


# pred_file="model_predictions/hvnguyen/mistral-nemo-minitron-8b-instruct-data-v7org-16x4/test_mistral-nemo-minitron-8b-instruct-data-v7-16x4-mimicsql_result.jsonl"
# echo "Evaluating $pred_file"
# python mimicsql_eval.py \
#     --pred_file "$pred_file" \
#     --db_path databases/mimic_all.db \
#     --num_workers -1 \
#     --timeout 60 \
#     --out_file outputs/hvnguyen/mistral-nemo-minitron-8b-instruct-data-v7org-16x4 \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------

# pred_file="model_predictions/hvnguyen/mistral-nemo-minitron-8b-instruct-data-v7org-16x4/test_mistral-nemo-minitron-8b-instruct-data-v7-16x4-ehrsql-eicu_result.jsonl"
# echo "Evaluating $pred_file"
# python ehrsql_eval.py \
#     --pred_file "$pred_file" \
#     --db_path databases/eicu.sqlite \
#     --num_workers 2 \
#     --timeout 60 \
#     --out_file outputs/hvnguyen/mistral-nemo-minitron-8b-instruct-data-v7org-16x4 \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------

# pred_file="model_predictions/hvnguyen/mistral-nemo-minitron-8b-instruct-data-v7org-16x4/test_mistral-nemo-minitron-8b-instruct-data-v7-16x4-ehrsql-mimiciii_result.jsonl"
# echo "Evaluating $pred_file"
# python ehrsql_eval.py \
#     --pred_file "$pred_file" \
#     --db_path databases/mimic_iii.sqlite \
#     --num_workers 2 \
#     --timeout 60 \
#     --out_file outputs/hvnguyen/mistral-nemo-minitron-8b-instruct-data-v7org-16x4 \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------


pred_file="model_predictions/hvnguyen/mistral-nemo-minitron-8b-instruct-data-v7org-16x4/test_mistral-nemo-minitron-8b-instruct-data-v7-16x4-vinmec_result.jsonl"
echo "Evaluating $pred_file"
python psql_sql_eval.py \
    --config configs/sql_eval.yaml \
    --pred_file "$pred_file" \
    --out_file outputs/hvnguyen/mistral-nemo-minitron-8b-instruct-data-v7org-16x4 \
    --input_template string
echo "Done\n" 
#----------------------------------------------------------------------------------------------------