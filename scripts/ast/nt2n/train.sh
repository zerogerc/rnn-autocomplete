#!/bin/bash

mkdir saved
mkdir saved/$1
PYTHONPATH=. python3 zerogercrnn/experiments/ast_level/nt2n/main.py \
    --title $1 \
    --train_file "data/ast/file_train.json" \
    --data_limit 10000 \
    --model_save_dir saved/$1 \
    --cuda \
    --real_data \
    --seq_len 50 \
    --batch_size 100 \
    --learning_rate 0.005 \
    --epochs 20 \
    --decay_after_epoch 0 \
    --decay_multiplier 0.9 \
    --weight_decay=0. \
    --hidden_size 1500 \
    --num_layers 1 \
    --dropout 0.01 \
    --non_terminals_num 97 \
    --non_terminal_embedding_dim 10 \
    --terminals_num 50001 \
    --terminal_embedding_dim 50 \
    --terminal_embeddings_file data/ast/terminal_embeddings.txt

