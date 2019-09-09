#!/bin/bash

export SUBMIT_DIR=`pwd`/..
echo "Will create submit files based on directory SUBMIT_DIR="$SUBMIT_DIR

if [[ "$1" == "resubmit" ]];
then
    echo "Preparing jobfiles to resubmit"
    python3 prepare_jobfiles_to_resubmit.py
    jobfiles_path="to_resubmit/jobfiles"
    \ls -1 to_resubmit/jobfiles/*.json | sed "s/to_resubmit\/jobfiles\///" | sed "s/\.json$//" > jobfiles.txt
else
    rm -Rf jobfiles jobfiles.txt jobfiles.tgz
    echo "Preparing jobfiles"
    python3 prepare_jobfiles.py
    jobfiles_path="jobfiles"
    \ls -1 jobfiles/*.json | sed "s/jobfiles\///" | sed "s/\.json$//" > jobfiles.txt
fi

echo "Preparing job chunks"

python chunk_submits.py 20 "$SUBMIT_DIR" "$jobfiles_path" > jobfiles_merged.txt

#Split on line, not on space
IFS=$'\n'

for f in `cat jobfiles_merged.txt`; do
    rm analyze_.sub
    cat template.sub > analyze_.sub
    echo "./hmm_analyze.sh $f" >> analyze_.sub
    qsub analyze_.sub

done
