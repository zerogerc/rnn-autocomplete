#!/bin/bash

mkdir saved
mkdir saved/$1
PYTHONPATH=. python3 zerogercrnn/experiments/js/ast_level/main/py_main.py \
    --title $1 \
    --config_file "zerogercrnn/experiments/js/ast_level/main/config.json" \
    --train_file "data/file_train.json" \
    --eval_file "data/file_eval.json" \
    --model_save_dir "saved" \
    --real_data
