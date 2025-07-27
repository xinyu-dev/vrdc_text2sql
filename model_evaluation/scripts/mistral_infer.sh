export PYTHONPATH=.

python mistral_infer.py \
    --task_name mistral-3-tasks \
    --checkpoint_path /home/jovyan/lustre/users/hvnguyen/experiments/nemo_1.0/mistral-3-tasks/checkpoints/hf \
    --max_seq_length 8192 \
    --batch_size 64 \
    --save_dir results \
    --dataset_path /home/jovyan/lustre/users/hvnguyen/data/nemo/text2sql/deepseek_r1_vinmec_generated_data/mistral_minitron_test.jsonl \
    --stop_token <extra_id_1>