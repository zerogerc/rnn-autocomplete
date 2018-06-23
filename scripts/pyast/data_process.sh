#!/bin/bash

PYTHONPATH=. python3 -m cProfile -o perf.prof scripts/ast/data_process.py \
    --file_train_raw "data/pyast/python10k_train.json" \
    --file_eval_raw "data/pyast/python5k_eval.json" \
    --file_non_terminals "data/pyast/non_terminals.json" \
    --file_terminals "data/pyast/terminals.json" \
    --file_train_converted "data/pyast/programs_training_seq.json" \
    --file_eval_converted "data/pyast/programs_eval_seq.json" \
    --file_train "data/pyast/file_train.json" \
    --file_eval "data/pyast/file_eval.json"