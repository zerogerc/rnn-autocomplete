#!/bin/bash

mkdir saved
mkdir saved/$2
mkdir eval
mkdir eval/$2

PYTHONPATH=. python3 zerogercrnn/experiments/ast_level/main.py \
    --title $2 \
    --prediction $1 \
    --eval_results_directory eval/$2 \
    --train_file "data/ast/file_train.json" \
    --data_limit 100000 \
    --model_save_dir saved/$2 \
    --seq_len 50 \
    --batch_size 80 \
    --learning_rate 0.001 \
    --epochs 30 \
    --decay_after_epoch 0 \
    --decay_multiplier 0.9 \
    --weight_decay=0. \
    --hidden_size 1500 \
    --num_layers 1 \
    --dropout 0.01 \
    --layered_hidden_size 300 \
    --non_terminals_num 97 \
    --non_terminal_embedding_dim 20 \
    --non_terminals_file "data/ast/non_terminals.json" \
    --non_terminal_embeddings_file "data/ast/non_terminal_embeddings.txt" \
    --terminals_num 50001 \
    --terminal_embedding_dim 100 \
    --terminals_file "data/ast/terminals.json" \
    --terminal_embeddings_file "data/ast/terminal_embeddings.txt" \
    --nodes_depths_stat_file "eval/ast/stat/node_depths.json"

