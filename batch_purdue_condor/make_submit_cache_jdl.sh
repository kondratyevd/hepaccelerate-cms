#!/bin/bash

export SUBMIT_DIR=`pwd`/..
echo "Will create submit files based on directory SUBMIT_DIR="$SUBMIT_DIR

#Clean old job files, copy from output directory
rm -Rf jobfiles jobfiles.txt jobfiles.tgz
echo "Preparing jobfiles"
python3 prepare_jobfiles.py

tar -cvzf jobfiles.tgz jobfiles
\ls -1 jobfiles/*.json | sed "s/jobfiles\///" | sed "s/\.json$//" > jobfiles.txt

echo "Preparing job chunks"
#Run 50 different random chunks per job
python chunk_submits.py 20 > jobfiles_merged.txt

#Prepare submit script
cat cache.jdl > submit_cache.jdl

#Split on line, not on space
IFS=$'\n'
for f in `cat jobfiles_merged.txt`; do
    echo "Arguments = --jobfiles "$f >> submit_cache.jdl
    echo "Queue" >> submit_cache.jdl
    echo >> submit_cache.jdl
done
#echo "Please run 'export SUBMIT_DIR=`pwd`/..'"

