#!/bin/bash

PYTHONPATH=. python3 scripts/ast/glove_tokens.py \
    --task terminals \
    --input_file data/programs_training.json \
    --output_file data/ast/glove_terminals_plain.txt \
    --token_map_file data/ast/terminals_map.json