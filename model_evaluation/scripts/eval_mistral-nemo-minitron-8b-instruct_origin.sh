#----------------------------------------------------------------------------------------------------
# Base models: mistral-nemo-minitron-8b-instruct
#----------------------------------------------------------------------------------------------------
export PYTHONPATH=.

pred_file="model_predictions/model_base/mistral-nemo-minitron-8b-instruct/mistral-nemo-minitron-8b-instruct_ehrsql_mimiciii_results.json"
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

pred_file="model_predictions/model_base/mistral-nemo-minitron-8b-instruct/mistral-nemo-minitron-8b-instruct_ehrsql_eicu_results.json"
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

pred_file="model_predictions/model_base/mistral-nemo-minitron-8b-instruct/mistral-nemo-minitron-8b-instruct_mimicsql_results.json"
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

pred_file="model_predictions/model_base/mistral-nemo-minitron-8b-instruct/mistral-nemo-minitron-8b-instruct_vinmec_results.json"
echo "Evaluating $pred_file"
python psql_sql_eval.py \
    --config configs/sql_eval.yaml \
    --pred_file "$pred_file" \
    --out_file results \
    --input_template chat
# all: exactly: 0.3796020679931067, subset: 0.3819520601597995
echo "Done\n" 
#----------------------------------------------------------------------------------------------------