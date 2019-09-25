#!/bin/bash

set -e

#Set some default arguments
export NTHREADS=24
export PYTHONPATH=coffea:hepaccelerate:. 
export HEPACCELERATE_CUDA=0
export KERAS_BACKEND=tensorflow
export NUMBA_NUM_THREADS=$NTHREADS
export OMP_NUM_THREADS=$NTHREADS

#This is where the skim files are loaded form
export CACHE_PATH=/depot/cms/hmm/cache_test_condor/

#Local output directory in worker node tmp
export OUTDIR=/depot/cms/hmm/out_$USER/

#Go to code directory
cd $SUBMIT_DIR

#Run the code
rm -f $CACHE_PATH/datasets.json
python3 tests/hmm/analysis_hmumu.py \
    --action cache \
    --nthreads $NTHREADS \
    --cache-location $CACHE_PATH \
    --datapath /mnt/hadoop/ \
    --maxchunks -1 --chunksize 1 \
    --out $OUTDIR\
    --jobfiles "$@"\
    --era 2016 --era 2017 --era 2018

echo "job done"
