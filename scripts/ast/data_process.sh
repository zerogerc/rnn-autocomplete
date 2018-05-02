#!/bin/bash

PYTHONPATH=. python3 -m cProfile -o perf.prof scripts/ast/data_process.py \
    --file_train_raw "data/programs_training.json" \
    --file_eval_raw "data/programs_eval.json" \
    --file_non_terminals "data/ast/non_terminals.json" \
    --file_terminals "data/ast/terminals.json" \
    --file_train_converted "data/ast/programs_training_seq.json" \
    --file_eval_converted "data/ast/programs_eval_seq.json" \
    --file_train "data/ast/file_train.json" \
    --file_eval "data/ast/file_eval.json" \
    --file_glove_map "data/ast/terminals_map.json" \
    --file_glove_vocab "data/ast/vocab.txt" \
    --file_glove_terminals "data/ast/glove_terminals.json" \
    --file_glove_non_terminals "data/ast/glove_non_terminals_corpus.txt"