#!/bin/bash

mkdir saved
mkdir saved/$1
PYTHONPATH=. python3 zerogercrnn/experiments/token_level/main.py \
    --title $1 \
    --task accuracy \
    --eval_file "data/tokens/file_eval.json" \
    --embeddings_file "data/tokens/vectors.txt" \
    --data_limit 10000 \
    --model_save_dir saved/$1 \
    --real_data \
    --tokens_count 51000 \
    --seq_len 50 \
    --batch_size 100 \
    --learning_rate 0.005 \
    --epochs 20 \
    --decay_after_epoch 0 \
    --decay_multiplier 0.9 \
    --embedding_size 50 \
    --hidden_size 1500 \
    --num_layers 1 \
    --dropout 0.01 \
    --weight_decay=0.
