#!/bin/bash

queue=${1:-"cms"}
perjob=${2:-20}

export SUBMIT_DIR=`pwd`/..
echo "Will create submit files based on directory SUBMIT_DIR="$SUBMIT_DIR

rm -Rf jobfiles jobfiles.txt jobfiles.tgz
echo "Preparing jobfiles"
python3 prepare_jobfiles.py
jobfiles_path="jobfiles"
\ls -1 jobfiles/*.json | sed "s/jobfiles\///" | sed "s/\.json$//" > jobfiles.txt

echo "Preparing job chunks"

python chunk_submits.py $perjob "$SUBMIT_DIR" "$jobfiles_path" > jobfiles_merged.txt

#Split on line, not on space
IFS=$'\n'

for f in `cat jobfiles_merged.txt`; do
    rm cache_.sub
    cat template.sub > cache_.sub
    echo "./hmm_cache.sh $f" >> cache_.sub
    qsub cache_.sub -q $queue
    break # to test on 1 job
done
