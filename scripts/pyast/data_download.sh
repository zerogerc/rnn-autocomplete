#!/bin/bash

wget 'http://files.srl.inf.ethz.ch/data/py150.tar.gz'
gunzip -c py150.tar.gz | tar xopf -
rm py150.tar.gz
mkdir data/pyast
mv python50k_eval.json data/pyast/python50k_eval.json
mv python100k_train.json data/pyast/python100k_train.json
cd ..