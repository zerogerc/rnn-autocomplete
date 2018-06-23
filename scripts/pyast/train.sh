#!/bin/bash

mkdir saved
mkdir saved/$2
mkdir eval
mkdir eval/$2

PYTHONPATH=. python3 -m cProfile -o program.prof zerogercrnn/experiments/ast_level/main.py \
    --title $2 \
    --prediction $1 \
    --eval_results_directory eval/$2 \
    --train_file "data/pyast/file_train.json" \
    --data_limit 100 \
    --model_save_dir saved/$2 \
    --seq_len 5 \
    --batch_size 5 \
    --learning_rate 0.001 \
    --epochs 30 \
    --decay_after_epoch 0 \
    --decay_multiplier 0.8 \
    --weight_decay=0. \
    --hidden_size 500 \
    --num_layers 1 \
    --dropout 0.01 \
    --layered_hidden_size 100 \
    --num_tree_layers 30 \
    --non_terminals_num 322 \
    --non_terminal_embedding_dim 50 \
    --non_terminals_file "data/pyast/non_terminals.json" \
    --terminals_num 50001 \
    --terminal_embedding_dim 50 \
    --terminals_file "data/pyast/terminals.json" \
    --node_depths_embedding_dim 20 \
    --nodes_depths_stat_file "eval/ast/stat/node_depths.json"

