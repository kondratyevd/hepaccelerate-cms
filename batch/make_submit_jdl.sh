#!/bin/bash

rm -Rf jobfiles jobfiles.txt jobfiles.tgz
cp -R ../out/jobfiles ./
tar -cvzf jobfiles.tgz jobfiles
\ls -1 jobfiles/*.json | sed "s/jobfiles\///" | sed "s/\.json$//" > jobfiles.txt
python submits.py > jobfiles_merged.txt

cat analyze.jdl > submit.jdl

IFS=$'\n'
for f in `cat jobfiles_merged.txt`; do
    echo "Arguments = "$f >> submit.jdl
    echo "Queue" >> submit.jdl
    echo >> submit.jdl
done
