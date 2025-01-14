#!/bin/bash

MAX_JOBS=300
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

python chunk_submits.py $perjob "$SUBMIT_DIR" "$jobfiles_path" "jobfiles.txt"> jobfiles_merged.txt

#Split on line, not on space
IFS=$'\n'

njobs=$(wc -l jobfiles_merged.txt | awk '{ print $1 }')
if [ $njobs -gt $MAX_JOBS ]; then
    echo "You are trying to create $njobs jobs, and the threshold is $MAX_JOBS. To override this, change the value of  MAX_JOBS in submit_analyse.sh"
    return
fi

for f in `cat jobfiles_merged.txt`; do
    rm analyze_.sub
    cat template.sub > analyze_.sub
    echo "./hmm_analyze.sh $f" >> analyze_.sub
    qsub analyze_.sub -q $queue

done
