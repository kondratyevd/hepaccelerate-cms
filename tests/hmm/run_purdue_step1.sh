#!/bin/bash
#Abort the script if any step fails
#set -e

#Use this many threads (max 4 makes sense, does not scale above due to numpy serialness)
export NTHREADS=4

#Set to -1 to run on all files, 1 for debugging/testing
export MAXCHUNKS=1

#This is where the intermediate analysis files will be saved and loaded from
#As long as one person produces it, other people can run the analysis on this

export CACHE_PATH=/tmp/dkondra/cache

export PYTHONPATH=coffea:hepaccelerate:.
export NUMBA_THREADING_LAYER=tbb
export NUMBA_ENABLE_AVX=1
export NUMBA_NUM_THREADS=$NTHREADS
export OMP_NUM_THREADS=$NTHREADS
export HEPACCELERATE_CUDA=0
export KERAS_BACKEND=tensorflow

#This is the location of the input NanoAOD and generally does not need to be changed

export INPUTDATAPATH=/mnt/hadoop/

## Step 1: cache ROOT data (need to repeat only when list of files or branches changes)
## This can take a few hours currently for the whole run (using maxchunks -1 and --nthreads 24)

python3 tests/hmm/analysis_hmumu.py \
   --action cache --maxchunks $MAXCHUNKS --chunksize 1 \
   --nthreads 24 --cache-location $CACHE_PATH \
   --datapath $INPUTDATAPATH --era 2016 #--era 2017 --era 2018

