#!/bin/bash

export CURRENT_DIR=$(pwd)
cd ../
export PYTHONPATH=coffea:hepaccelerate:.

#Run merge
python3 tests/hmm/analysis_hmumu.py \
    --action merge \
    --nthreads 8 \
    --out /depot/cms/hmm/out_$USER

#Run plots
python3 tests/hmm/plotting.py --input /depot/cms/hmm/out_$USER --nthreads 8

cd $CURRENT_DIR
