#!/bin/bash
#Abort the script if any step fails
set -e

#Set NTHREADS=24 to run on whole machine, =1 for debugging
NTHREADS=20

#Set to -1 to run on all files, 1 for debugging
MAXFILES=-1

#This is where the intermediate analysis files will be saved and loaded from
#As long as one person produces it, other people can run the analysis on this

export CACHE_PATH=/storage/user/$USER/hmm/cache
#export CACHE_PATH=/nvme1/jpata/cache

export SINGULARITY_IMAGE=/storage/user/jpata/cupy.simg

if [ ! -f "$SINGULARITY_IMAGE" ]; then
    echo "Singularity image is missing, check the script"
    exit 1
fi


if [ ! -d "$CACHE_PATH" ]; then
    echo "Cache path is missing, check the script"
    exit 1
fi

export PYTHONPATH=coffea:hepaccelerate:.
export NUMBA_THREADING_LAYER=tbb
export NUMBA_ENABLE_AVX=0
export NUMBA_NUM_THREADS=$NTHREADS
export OMP_NUM_THREADS=$NTHREADS
export HEPACCELERATE_CUDA=0

## Step 1: cache ROOT data (need to repeat only when list of files or branches changes)
## This can take a few hours currently for the whole run (using MAXFILES=-1 and NTHREADS=24)
singularity exec --nv -B /storage -B /nvme1 $SINGULARITY_IMAGE python3 tests/hmm/analysis_hmumu.py \
    --action cache --maxfiles $MAXFILES --chunksize 1 \
    --nthreads $NTHREADS --cache-location $CACHE_PATH \
    --datapath /storage/user/jpata/ --era 2016 --era 2017 --era 2018
