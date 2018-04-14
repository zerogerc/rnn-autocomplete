#!/bin/bash

mkdir data
cd data
wget 'http://files.srl.inf.ethz.ch/data/js_dataset.tar.gz'
gunzip -c js_dataset.tar.gz | tar xopf -
gunzip -c data.tar.gz | tar xopf -
rm js_dataset.tar.gz
rm data.tar.gz
mkdir ast
mkdir token
cd ..