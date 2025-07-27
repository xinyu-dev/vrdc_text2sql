#----------------------------------------------------------------------------------------------------
# Base models: Gwen/QwQ-32B
#----------------------------------------------------------------------------------------------------
export PYTHONPATH=.


pred_file="model_predictions/model_base/QwQ-32B/qwq-32b_ehrsql_eicu_results.json"
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

pred_file="model_predictions/model_base/QwQ-32B/qwq-32b_ehrsql_mimiciii_results.json"
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

pred_file="model_predictions/model_base/QwQ-32B/qwq-32b_mimicsql_results.json"
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

pred_file="model_predictions/model_base/QwQ-32B/qwq-32b_vinmec_results.json"
echo "Evaluating $pred_file"
python psql_sql_eval.py \
    --config configs/sql_eval.yaml \
    --pred_file "$pred_file" \
    --out_file results \
    --input_template chat
# all: exactly: 0.5174274464051335, subset: 0.5254484468426426
echo "Done\n" 
#----------------------------------------------------------------------------------------------------