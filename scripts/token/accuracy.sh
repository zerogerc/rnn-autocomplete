#!/bin/bash

PYTHONPATH=. python3 zerogercrnn/experiments/token_level/results.py \
    --metrics accuracy \
    --saved_model saved/$1 \
    --eval_file "data/tokens/file_eval.json" \
    --cuda \
    --tokens_count 51000 \
    --seq_len 50 \
    --batch_size 100 \
    --embedding_size 50 \
    --hidden_size 1500 \
    --num_layers 1 \
    --dropout 0.01 \
    --weight_decay=0.
