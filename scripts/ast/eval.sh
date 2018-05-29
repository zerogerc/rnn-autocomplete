#!/bin/bash
#!/bin/bash

mkdir saved
mkdir saved/$2
mkdir eval
mkdir eval/$2

PYTHONPATH=. python3 zerogercrnn/experiments/ast_level/main.py \
    --title temp \
    --saved_model "saved/26May_nt2n_base_attention_plus_layered_hs500/model_epoch_30" \
    --prediction nt2n_base_attention_plus_layered \
    --eval_file "data/ast/file_eval.json" \
    --eval_results_directory "eval/temp" \
    --data_limit 50000 \
    --model_save_dir saved/temp \
    --eval \
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
    --layered_hidden_size 500 \
    --non_terminals_num 97 \
    --non_terminal_embedding_dim 50 \
    --non_terminals_file "data/ast/non_terminals.json" \
    --non_terminal_embeddings_file "data/ast/non_terminal_embeddings.txt" \
    --terminals_num 50001 \
    --terminal_embedding_dim 50 \
    --terminal_embeddings_file "data/ast/terminal_embeddings.txt"
