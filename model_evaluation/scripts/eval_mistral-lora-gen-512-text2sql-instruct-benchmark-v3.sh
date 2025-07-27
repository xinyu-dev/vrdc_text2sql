#----------------------------------------------------------------------------------------------------
# Finetuned models: mistral-nemo-minitron-8b-instruct (single task - loRa - Remove EHR-2024-MIMIC-IV dataset- tokens to generate: 512)
#----------------------------------------------------------------------------------------------------
export PYTHONPATH=.

pred_file="model_predictions/test_mistral-lora-gen-512-text2sql-instruct-benchmark-mimicsql-v3_result.jsonl"
echo "Evaluating $pred_file"
python ehrsql_eval.py \
    --pred_file "$pred_file" \
    --db_path databases/mimic_all.db \
    --num_workers -1 \
    --timeout 60 \
    --out_file outputs \
    --ndigits 2 
echo "Done\n" 
#----------------------------------------------------------------------------------------------------
pred_file="model_predictions/test_mistral-lora-gen-512-text2sql-instruct-benchmark-ehrsql-mimic-v3_result.jsonl"
echo "Evaluating $pred_file"
python ehrsql_eval.py \
    --pred_file "$pred_file" \
    --db_path databases/mimic_iii.sqlite \
    --num_workers -1 \
    --timeout 60 \
    --out_file outputs \
    --ndigits 2 
echo "Done\n" 
#----------------------------------------------------------------------------------------------------
pred_file="model_predictions/test_mistral-lora-gen-512-text2sql-instruct-benchmark-ehrsql-eicu-v3_result.jsonl"
echo "Evaluating $pred_file"
python ehrsql_eval.py \
    --pred_file "$pred_file" \
    --db_path databases/eicu.sqlite \
    --num_workers -1 \
    --timeout 60 \
    --out_file outputs \
    --ndigits 2 
echo "Done\n" 
#----------------------------------------------------------------------------------------------------

pred_file="model_predictions/test_mistral-lora-gen-512-text2sql-instruct-benchmark-vinmec_result.jsonl"
echo "Evaluating $pred_file"
python psql_sql_eval.py \
    --config configs/sql_eval.yaml \
    --pred_file "$pred_file" \
    --out_file results \
    --input_template string
# all: exactly: 0.5174274464051335, subset: 0.5254484468426426
echo "Done\n" 
#----------------------------------------------------------------------------------------------------