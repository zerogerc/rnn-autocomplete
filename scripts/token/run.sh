#!/bin/bash

mkdir saved
mkdir saved/$1
PYTHONPATH=. python3 zerogercrnn/experiments/token_level/main.py \
    --title $1
    --train_file "data/tokens/file_train.json"
    --eval_file "data/tokens/file_eval.json"
    --embeddings_file "data/tokens/vector.txt"
    --model_save_dir saved/$1
    --cuda
    --real_data
    --tokens_count 51000
    --seq_len 50
    --batch_size 400
    --learning_rate 0.005
    --epochs 20
    --decay_after_epoch 0
    --decay_multiplier 0.9
    --embedding_size 50
    --hidden_size 1500
    --num_layers 1
    --dropout 0.01
    --weight_decay=0.
