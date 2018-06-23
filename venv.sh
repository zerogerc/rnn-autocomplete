#!/usr/bin/env bash

#!/bin/bash

VENV_DIR=./env

if [ ! -d $VENV_DIR ]; then
    echo 'creating venv...'
    python3 -m venv $VENV_DIR
    echo 'done'
else
    echo 'venv have been already created'
fi

source $VENV_DIR/bin/activate
echo 'venv activated'

echo 'installing standalone requirements...'
echo "$(command -v nvidia-smi)"
if [ -x "$(command -v nvidia-smi)" ]; then
    pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
else
    pip3 install http://download.pytorch.org/whl/torch-0.4.0-cp36-cp36m-macosx_10_7_x86_64.whl
fi

echo 'done'

echo 'updating requirements...'
pip3 install -r requirements.txt
echo 'done'