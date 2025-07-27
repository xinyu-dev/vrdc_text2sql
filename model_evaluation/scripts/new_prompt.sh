# -------------- eicu ----------------
python ehrsql_eval.py \
    --pred_file model_evaluation/model_predictions/new-prompt/test_mistral-text2sql-ehrsql-eicu_result.jsonl \
    --db_path databases/eicu.sqlite \
    --out_file outputs 

pred_file="/localhome/local-hndo/model-evaluation/model_evaluation/model_predictions/rag/mistral-text2sql-v28/test_rag_vllm_eicu_result_mis_embedd.jsonl"
echo "Evaluating $pred_file"
python ehrsql_eval.py \
    --pred_file "$pred_file" \
    --db_path databases/eicu.sqlite \
    --num_workers 70 \
    --timeout 60 \
    --out_file outputs/rag/mistral-text2sql-v28 \
    --ndigits 2 
echo "Done\n" 

pred_file="/Users/hndo/Library/CloudStorage/OneDrive-NVIDIACorporation/Documents/model-evaluation/model_evaluation/model_predictions/new-ddl-inst/qwq-32b/QwQ-32B_ehrsql_eicu_result.jsonl"
echo "Evaluating $pred_file"
python ehrsql_eval_hndo.py \
    --pred_file "$pred_file" \
    --db_path databases/eicu.sqlite \
    --num_workers 2 \
    --timeout 120 \
    --out_file outputs/new-ddl-inst/qwq-32b \
    --ndigits 2 
echo "Done\n" 

# -------------- mimiciii -------------
python ehrsql_eval.py \
    --pred_file model_evaluation/model_predictions/new-prompt/test_mistral-text2sql-ehrsql-mimic_result.jsonl \
    --db_path databases/mimic_iii.sqlite \
    --out_file outputs 

pred_file="/localhome/local-hndo/model-evaluation/model_evaluation/model_predictions/rag/mistral-text2sql-v28/test_rag_vllm_ehrsql_mimiciii_result_mis_embedd.jsonl"
echo "Evaluating $pred_file"
python ehrsql_eval.py \
    --pred_file "$pred_file" \
    --db_path databases/mimic_iii.sqlite \
    --num_workers 70 \
    --timeout 60 \
    --out_file outputs/rag/mistral-text2sql-v28 \
    --ndigits 2 
echo "Done\n" 

pred_file="/localhome/local-hndo/hndo_eval/model_evaluation/model_predictions/new-ddl-inst/medgemma-4b-pt/medgemma-4b-pt_ehrsql_mimiciii_result.jsonl"
echo "Evaluating $pred_file"
python ehrsql_eval_hndo.py \
    --pred_file "$pred_file" \
    --db_path databases/mimic_iii.sqlite \
    --num_workers 2 \
    --timeout 120 \
    --out_file outputs/new-ddl-inst/medgemma-4b-pt \
    --ndigits 2 
echo "Done\n" 

# -------------- mimicsql --------------
python ehrsql_eval.py \
    --pred_file model_evaluation/model_predictions/new-prompt/test_mistral-text2sql-mimicsql_result.jsonl \
    --db_path databases/mimic_all.db \
    --out_file outputs 

pred_file="/localhome/local-hndo/hndo_eval/model_evaluation/model_predictions/hvnguyen/mistral-nemo-minitron-8b-instruct-dv8-pv2-24x4/test_mistral-nemo-minitron-8b-instruct-dv8-pv2-24x4-mimicsql_result.jsonl"
echo "Evaluating $pred_file"
python mimicsql_eval.py \
    --pred_file "$pred_file" \
    --db_path databases/mimic_all_lower.db \
    --num_workers 70 \
    --timeout 60 \
    --out_file outputs/hvnguyen/mistral-nemo-minitron-8b-instruct-dv8-pv2-24x4 \
    --ndigits 2 
echo "Done\n" 

pred_file="/localhome/local-hndo/model-evaluation/model_evaluation/model_predictions/rag/mistral-text2sql-v28/test_rag_vllm_mimicsql_result_mis_embedd.jsonl"
echo "Evaluating $pred_file"
python mimicsql_eval.py \
    --pred_file "$pred_file" \
    --db_path databases/mimic_all.db \
    --num_workers 70 \
    --timeout 60 \
    --out_file outputs/rag/mistral-text2sql-v28 \
    --ndigits 2 
echo "Done\n" 

# -------------- vinmec --------------

pred_file="/Users/hndo/Library/CloudStorage/OneDrive-NVIDIACorporation/Documents/hndo_eval/model_evaluation/model_predictions/rag/mistral-text2sql-v28-rerun2/test_rag_vllm_vinmec_result_mis_embedd.jsonl"
echo "Evaluating $pred_file"
python psql_sql_eval.py \
    --config configs/sql_eval.yaml \
    --pred_file "$pred_file" \
    --out_file outputs/rag/mistral-text2sql-v28-rerun2-pp \
    --input_template string
echo "Done\n"   