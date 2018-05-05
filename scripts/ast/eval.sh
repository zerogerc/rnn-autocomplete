#!/bin/bash

mkdir saved
mkdir saved/$2
PYTHONPATH=. python3 zerogercrnn/experiments/ast_level/main.py \
    --title $2 \
    --prediction $1 \
    --eval_file "data/ast/file_eval.json" \
    --data_limit 10000 \
    --model_save_dir saved/$2 \
    --saved_model $3 \
    --cuda \
    --eval \
    --seq_len 50 \
    --batch_size 80 \
    --learning_rate 0.005 \
    --epochs 20 \
    --decay_after_epoch 0 \
    --decay_multiplier 0.9 \
    --weight_decay=0. \
    --hidden_size 1000 \
    --num_layers 1 \
    --dropout 0.01 \
    --non_terminals_num 97 \
    --non_terminal_embedding_dim 10 \
    --terminals_num 50001 \
    --terminal_embedding_dim 100 \
    --terminal_embeddings_file data/ast/terminal_embeddings.txt

