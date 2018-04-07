#!/bin/bash

mkdir data
cd data
wget 'http://files.srl.inf.ethz.ch/data/js_dataset.tar.gz'
gunzip -c js_dataset.tar.gz | tar xopf -
cd ..