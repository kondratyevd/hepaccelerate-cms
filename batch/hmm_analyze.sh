#!/bin/bash

set -e

ls /storage

env

workdir=`pwd`
tar xf jobfiles.tgz

export JOBFILE=$1
export NTHREADS=8
export PYTHONPATH=coffea:hepaccelerate:. 
export HEPACCELERATE_CUDA=0
export KERAS_BACKEND=tensorflow

export CACHE_PATH=/storage/user/$USER/hmm/cache
export NUMBA_NUM_THREADS=$NTHREADS
export OMP_NUM_THREADS=$NTHREADS
export OUTDIR=out

cd /data/jpata/hmumu/hepaccelerate-cms/

python3 tests/hmm/analysis_hmumu.py \
    --action analyze \
    --nthreads $NTHREADS --cache-location $CACHE_PATH \
    --datapath /storage/user/jpata/ \
    --out $workdir/$OUTDIR \
    --jobfile $workdir/jobfiles/$JOBFILE.json


cp -R $workdir/$OUTDIR /storage/user/$USER/hmm/
ls $CACHE_PATH
echo "job done"
