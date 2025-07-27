export PYTHONPATH=$PYTHONPATH:$(pwd)

python mimicsql_eval.py \
    --db_file databases/mimic_iii.sqlite \
    --lookup_file databases/lookup.json \
    --output_file model_predictions/test_mistral-3-tasks-mimicsql_result.jsonl
