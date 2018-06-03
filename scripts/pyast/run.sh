#!/bin/bash

./scripts/pyast/data_download.sh
./sripts/pyast/data_process.sh
./scripts/pyast/train.sh nt2n_base_attention_plus_layered 03Jun_py_nt2n_base_attention_plus_layered_hs1500_lhs500