##################
Bachelor's grad work in neural code completion
##################

Initial set up
=================
Create virtual environment: ``./venv.sh``

Activate virtual environment: ``source env/bin/activate``

Proposed models are working with AST so there is a possibility to complete any language. For now there is possibility to test model on two datasets:

1. Javascript (`js150 dataset link <https://www.sri.inf.ethz.ch/js150.php>`_)
2. Python (`py150 dataset link <https://www.sri.inf.ethz.ch/py150.php>`_)


Javascript
==============
To train model on Javascript dataset:

1. Download data: ``./scripts/ast/data_download.sh``
2. Process data: ``./scripts/ast/data_process.sh``
3. Train model: ``./scripts/ast/run.sh``

To change model parameters edit file: ``scripts/ast/train.sh``

Python
==============
To train model on Python dataset:

1. Download data: ``./scripts/pyast/data_download.sh``
2. Process data: ``./scripts/pyast/data_process.sh``
3. Train model: ``./scripts/pyast/run.sh``

To change model parameters edit file: ``scripts/pyast/train.sh``

Results
=============
For accuracy visualization tensorboard is used. To run it use: ``./scripts/tensorboard.sh``
