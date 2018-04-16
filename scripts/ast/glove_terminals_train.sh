#!/bin/bash

./build/vocab_count -max-vocab 50000 -min-count 10 < corpus.txt > vocab.txt
./build/cooccur -vocab-file vocab.txt -memory 1 < corpus.txt > cooccurrence.bin
./build/shuffle -memory 1 < cooccurrence.bin > cooccurrence.shuf.bin
./build/glove -input-file cooccurrence.shuf.bin -vocab-file vocab.txt -save-file vectors -verbose 0 \
    -vector-size 100 -threads 4 -alpha 0.75 -x-max 100.0 -eta 0.05 -binary 2 -model 2 -iter 1000 -checkpoint-every 100