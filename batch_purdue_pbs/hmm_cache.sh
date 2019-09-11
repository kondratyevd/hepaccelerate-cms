#!/bin/bash

set -e

#Set some default arguments
export NTHREADS=2
export PYTHONPATH=coffea:hepaccelerate:. 
export HEPACCELERATE_CUDA=0
export KERAS_BACKEND=tensorflow
export NUMBA_NUM_THREADS=$NTHREADS
export OMP_NUM_THREADS=$NTHREADS
export MAXCHUNKS=-1

#This is where the skim files are loaded form
export CACHE_PATH=/depot/cms/hmm/cache_test/

#Local output director in worker node tmp
export OUTDIR=/depot/cms/hmm/out_$USER/

#Go to code directory
cd $SUBMIT_DIR

export SCRAM_ARCH=slc7_amd64_gcc700
source /cvmfs/cms.cern.ch/cmsset_default.sh
eval `scramv1 runtime -sh`

#Run the code
python3 tests/hmm/analysis_hmumu.py \
    --action cache \
    --nthreads $NTHREADS --cache-location $CACHE_PATH \
    --datapath /mnt/hadoop \
    --out $OUTDIR \
    --jobfiles "$@"\
    --maxchunks $MAXCHUNKS\
    --chunksize 1

echo "job done"
