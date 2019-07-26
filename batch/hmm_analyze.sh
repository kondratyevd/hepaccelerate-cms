#!/bin/bash

set -e

ls /storage

env

workdir=`pwd`

export USER=$1
export NTHREADS=32
export PYTHONPATH=coffea:hepaccelerate:. 
export HEPACCELERATE_CUDA=0
export KERAS_BACKEND=tensorflow

export CACHE_PATH=/storage/user/$USER/hmm/cache
export NUMBA_NUM_THREADS=$NTHREADS
export OMP_NUM_THREADS=$NTHREADS
export MAXFILES=-1

cd /data/jpata/hmumu/hepaccelerate-cms/

python3 tests/hmm/analysis_hmumu.py \
    --action analyze --maxfiles $MAXFILES --chunksize 1 \
    --nthreads $NTHREADS --cache-location $CACHE_PATH \
    --datapath /storage/user/jpata/ --era 2016 --era 2017 --era 2018 \
    --dataset dy --out $workdir/out

cp -R $workdir/out /storage/user/$USER/hmm/out
ls $CACHE_PATH
echo "job done"
