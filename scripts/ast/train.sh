#!/bin/bash

mkdir saved
mkdir saved/$2
PYTHONPATH=. python3 zerogercrnn/experiments/ast_level/main.py \
    --title $2 \
    --prediction $1 \
    --train_file "data/ast/file_train.json" \
    --data_limit 100000 \
    --saved_model "saved/10May_nt2n_layered_attention_hs1500_shs_300_no_glove/model_epoch_30" \
    --model_save_dir saved/$2 \
    --seq_len 50 \
    --batch_size 80 \
    --learning_rate 0.0001 \
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
    --non_terminal_embeddings_file data/ast/non_terminal_embeddings.txt \
    --terminals_num 50001 \
    --terminal_embedding_dim 100 \
    --terminal_embeddings_file data/ast/terminal_embeddings.txt

