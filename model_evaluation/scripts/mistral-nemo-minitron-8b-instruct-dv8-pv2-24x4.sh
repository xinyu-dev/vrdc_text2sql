#----------------------------------------------------------------------------------------------------
# models: Mistral-Nemo-Minitron-8B-Instruct, data version: v8, prompt version: pv2, resource: gpu 24x4 
#----------------------------------------------------------------------------------------------------
export PYTHONPATH=.

PRED_ROOT_PATH="model_predictions/hvnguyen/mistral-nemo-minitron-8b-instruct-dv8-pv2-24x4"
OUT_ROOT_PATH="outputs/hvnguyen/mistral-nemo-minitron-8b-instruct-dv8-pv2-24x4"

mkdir -p ${OUT_ROOT_PATH}

# pred_file="${PRED_ROOT_PATH}/test_mistral-nemo-minitron-8b-instruct-dv8-pv2-24x4-mimicsql_result.jsonl"
# echo "Evaluating $pred_file"
# python mimicsql_eval.py \
#     --pred_file "$pred_file" \
#     --db_path databases/mimic_all.db \
#     --num_workers -1 \
#     --timeout 60 \
#     --out_file ${OUT_ROOT_PATH} \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------

# pred_file="${PRED_ROOT_PATH}/test_mistral-nemo-minitron-8b-instruct-dv8-pv2-24x4-ehrsql-eicu_result.jsonl"
# echo "Evaluating $pred_file"
# python ehrsql_eval.py \
#     --pred_file "$pred_file" \
#     --db_path databases/eicu.sqlite \
#     --num_workers 2 \
#     --timeout 60 \
#     --out_file ${OUT_ROOT_PATH} \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------

# pred_file="${PRED_ROOT_PATH}/test_mistral-nemo-minitron-8b-instruct-dv8-pv2-24x4-ehrsql-mimiciii_result.jsonl"
# echo "Evaluating $pred_file"
# python ehrsql_eval.py \
#     --pred_file "$pred_file" \
#     --db_path databases/mimic_iii.sqlite \
#     --num_workers 2 \
#     --timeout 60 \
#     --out_file ${OUT_ROOT_PATH} \
#     --ndigits 2 
# echo "Done\n" 
# #----------------------------------------------------------------------------------------------------


pred_file="${PRED_ROOT_PATH}/test_mistral-nemo-minitron-8b-instruct-dv8-pv2-24x4-vinmec_result.jsonl"
echo "Evaluating $pred_file"
python psql_sql_eval.py \
    --config configs/sql_eval.yaml \
    --pred_file "$pred_file" \
    --out_file ${OUT_ROOT_PATH} \
    --input_template string
echo "Done\n" 
# ----------------------------------------------------------------------------------------------------