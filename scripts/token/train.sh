#!/bin/bash

mkdir saved
mkdir saved/$2
mkdir eval
mkdir eval/$2

PYTHONPATH=. python3 -m cProfile -o program.prof zerogercrnn/experiments/token_level/main.py \
    --title $2 \
    --prediction $1 \
    --eval_results_directory eval/$2 \
    --train_file "data/tokens/file_train.json" \
    --data_limit 100000 \
    --model_save_dir saved/$2 \
    --seq_len 50 \
    --batch_size 80 \
    --learning_rate 0.001 \
    --epochs 30 \
    --decay_after_epoch 0 \
    --decay_multiplier 0.9 \
    --weight_decay=0. \
    --hidden_size 500 \
    --num_layers 1 \
    --dropout 0.01 \
    --tokens_num 51000 \
    --token_embedding_dim 50
