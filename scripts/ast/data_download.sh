#!/bin/bash

wget 'http://files.srl.inf.ethz.ch/data/js_dataset.tar.gz'
gunzip -c js_dataset.tar.gz | tar xopf -
rm js_dataset.tar.gz
rm data.tar.gz
rm programs_training.txt
rm programs_eval.txt
rm README.txt
mkdir data
mkdir data/ast
mv programs_training.json data/ast/programs_training.json
mv programs_eval.json data/ast/programs_eval.json
