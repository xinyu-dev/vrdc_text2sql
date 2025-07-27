#----------------------------------------------------------------------------------------------------
# Base models: llama3-sqlcoder=8b
#----------------------------------------------------------------------------------------------------
export PYTHONPATH=.


pred_file="model_predictions/model_base/llama3-sqlcoder-8b/llama3-sqlcoder-8b_ehrsql_eicu_results.json"
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

pred_file="model_predictions/model_base/llama3-sqlcoder-8b/llama3-sqlcoder-8b_ehrsql_mimiciii_results.json"
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

pred_file="model_predictions/model_base/llama3-sqlcoder-8b/llama3-sqlcoder-8b_mimicsql_results.json"
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

pred_file="model_predictions/model_base/llama3-sqlcoder-8b/llama3-sqlcoder-8b_vinmec_results.json"
echo "Evaluating $pred_file"
python psql_sql_eval.py \
    --config configs/sql_eval.yaml \
    --pred_file "$pred_file" \
    --out_file results \
    --input_template string
# all: exactly: 0.29219570001692907, subset: 0.29219570001692907
echo "Done\n" 
#----------------------------------------------------------------------------------------------------