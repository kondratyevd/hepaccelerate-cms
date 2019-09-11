#!/bin/bash

queue=${1:-"cms"}
perjob=${2:-20}

export SUBMIT_DIR=`pwd`/..
echo "Will create submit files based on directory SUBMIT_DIR="$SUBMIT_DIR

rm -Rf to_resubmit/*
echo "Preparing jobfiles"
python3 prepare_jobfiles.py --to_resubmit
jobfiles_path="to_resubmit/jobfiles"
\ls -1 to_resubmit/jobfiles/*.json | sed "s/to_resubmit\/jobfiles\///" | sed "s/\.json$//" > re_jobfiles.txt

echo "Preparing job chunks"

python chunk_submits.py $perjob "$SUBMIT_DIR" "$jobfiles_path" > re_jobfiles_merged.txt

#Split on line, not on space
IFS=$'\n'

for f in `cat re_jobfiles_merged.txt`; do
    rm cache_.sub
    cat template.sub > cache_.sub
    echo "./hmm_cache.sh $f" >> cache_.sub
    qsub cache_.sub -q $queue
#    break # to test on 1 job
done
