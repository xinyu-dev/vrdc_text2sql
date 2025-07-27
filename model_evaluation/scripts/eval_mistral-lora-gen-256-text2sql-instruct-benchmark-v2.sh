#----------------------------------------------------------------------------------------------------
# models: Mistral-Nemo-Minitron-8B-Instruct LoRA, tokens to generate: 256
#----------------------------------------------------------------------------------------------------
export PYTHONPATH=.


pred_file="model_predictions/test_mistral-lora-text2sql-instruct-benchmark-mimicsql_result.jsonl"
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

pred_file="model_predictions/test_mistral-lora-text2sql-instruct-benchmark-ehrsql-eicu_result.jsonl"
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

pred_file="model_predictions/test_mistral-lora-text2sql-instruct-benchmark-ehrsql-mimic_result.jsonl"
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


pred_file="model_predictions/test_mistral-lora-text2sql-instruct-benchmark-vinmec_result.jsonl"
echo "Evaluating $pred_file"
python psql_sql_eval.py \
    --config configs/sql_eval.yaml \
    --pred_file "$pred_file" \
    --out_file results \
    --input_template string
echo "Done\n" 
#----------------------------------------------------------------------------------------------------