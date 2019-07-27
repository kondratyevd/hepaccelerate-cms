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
export OUTDIR=out

echo "numjob", $NUMJOB

cd /data/jpata/hmumu/hepaccelerate-cms/

python3 tests/hmm/analysis_hmumu.py \
    --action analyze --maxfiles $MAXFILES --chunksize 5 \
    --nthreads $NTHREADS --cache-location $CACHE_PATH \
    --datapath /storage/user/jpata/ --era 2018 \
    --dataset data --dataset dy --dataset vbf --do-factorized-jec \
    --out $workdir/$OUTDIR

cp -R $workdir/$OUTDIR /storage/user/$USER/hmm/$OUTDIR
ls $CACHE_PATH
echo "job done"
