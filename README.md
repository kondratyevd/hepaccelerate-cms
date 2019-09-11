[![Build Status](https://travis-ci.com/jpata/hepaccelerate.svg?branch=master)](https://travis-ci.com/jpata/hepaccelerate-cms)
[![pipeline status](https://gitlab.cern.ch/jpata/hepaccelerate-cms/badges/master/pipeline.svg)](https://gitlab.cern.ch/jpata/hepaccelerate-cms/commits/master)

# hepaccelerate-cms

CMS-specific accelerated analysis code based on the [hepaccelerate](https://github.com/jpata/hepaccelerate) library.

~~~
#Installation
pip3 install --user scipy awkward uproot numba cffi lz4 cloudpickle pyyaml pylzma pandas
pip3.6 install --user backports.lzma

source /cvmfs/cms.cern.ch/cmsset_default.sh
mkdir ~/hepaccelerate/
cd ~/hepaccelerate/
cmsrel CMSSW_10_2_9
cd CMSSW_10_2_9/src
cmsenv

git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit

git clone https://github.com/kondratyevd/hepaccelerate-cms.git
cd hepaccelerate-cms
git submodule init
git submodule update

#Compile the C++ helper code (Rochester corrections and lepton sf, ROOT is needed)

cd tests/hmm/
make
cd ../..
~~~

Run the framework locally:

~~~
# Step 1: cache
. tests/hmm/run_purdue_step1.sh
# Step 2: analyze
. tests/hmm/run_purdue_step2.sh
# Step 3: produce plots and datacards
. tests/hmm/plots.sh out/
~~~

How to submit jobs:
===

The submission scripts `submit_cache.sh` and `submit_analyze.sh` can be ran with two arguments, for example `. submit_analyze.sh cms 20`.
The first argument is the name of PBS queue ("cms" by default), and the second argument is number of chunks per job (20 by default).

Full analysis (2016+17+18) contains about 4000 chunks, so by default ~200 jobs will be created.

Run these commands from Hammer frontend (hammer.rcac.purdue.edu):

Stage 1: caching
---

Cache is not meant to be produced frequently; reproducing it takes a lot of time and storage space. 
Please proceed to the next stage, unless you are confident that you need to reproduce the cache.

We currently store cache at `/depot/cms/hmm/cache/`, it is available for all members of the group.

~~~
# The input and output directories are defined in `hmm_cache.py`

cd batch_purdue_pbs
mkdir logs
. setup_proxy.sh

. submit_cache.sh
# wait for completion..
~~~
Once the jobs are completed, check how many failed:
~~~
python3 find_failed.py

# If any files in cache are missing, resubmit the jobs for the corresponding datasets:

. resubmit_cache.sh
~~~
*Hint:* depending on how many chunks are missing from the output, you may want to set different number of chunks per job for resubmission for faster processing (default is 20). 
For example, if 200 chunks are missing, you can run   
`. resubmit_cache.sh cms 1`    
to submit 200 small jobs (instead of 10 bigger jobs submitted by default). 

Repeat resubmision as many times as needed, and wait for all jobs to finish...

~~~
cp jobfiles.json /path/to/cache/
~~~

Stage 2 & 3: analysis, plots and datacards 
---

~~~
# The input and output directories are defined in `hmm_analyze.py`

cd batch_purdue_pbs
mkdir logs

. submit_analyze.sh
# wait for completion..

# Merge results and produce plots
. post.sh

~~~

<!---
Best results can be had if the CMS data is stored locally on a filesystem (few TB needed) and if you have a cache disk on the analysis machine of a few hundred GB.

A prebuilt singularity image with the GPU libraries is also provided: [link](http://login-1.hep.caltech.edu/~jpata/cupy.simg)


## Installation on Caltech T2 or GPU machine

On Caltech, an existing singularity image can be used to get the required python libraries.
~~~
git clone https://github.com/jpata/hepaccelerate-cms.git
cd hepaccelerate-cms
git checkout dev-aug-w2
git submodule init
git submodule update

#Compile the C++ helpers
cd tests/hmm
singularity exec /storage/user/jpata/cupy2.simg make -j4
cd ../..

#Run the code as a small test (small dataset by default, edit the file to change this)
#This should take approximately 5 minutes and processes 1 file from each dataset for each year
./tests/hmm/run.sh
~~~

## Running on full dataset using batch queue
We use the condor batch queue on Caltech T2 to run the analysis. It takes about 2-3h for all 3 years using factorized JEC. Without factorized JEC (using total JEC), the runtime is about 10 minutes.

~~~
#Submit batch jobs after this step is successful
mkdir /storage/user/$USER/hmm
export SUBMIT_DIR=`pwd`
cd batch
./make_submit_jdl.sh
condor_submit submit.jdl

... (wait for completion)
condor_submit merge.jdl

cd ..

#when all was successful, delete partial results
rm -Rf /storage/user/$USER/hmm/out/partial_results
du -csh /storage/user/$USER/hmm/out
~~~

# Misc notes
Luminosity, details on how to set up on this [link](https://cms-service-lumi.web.cern.ch/cms-service-lumi/brilwsdoc.html).
~~~
export PATH=$HOME/.local/bin:/cvmfs/cms-bril.cern.ch/brilconda/bin:$PATH
brilcalc lumi -c /cvmfs/cms.cern.ch/SITECONF/local/JobConfig/site-local-config.xml \
    -b "STABLE BEAMS" --normtag=/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json \
    -u /pb --byls --output-style csv -i /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/ReReco/Final/Cert_271036-284044_13TeV_23Sep2016ReReco_Collisions16_JSON.txt > lumi2016.csv

brilcalc lumi -c /cvmfs/cms.cern.ch/SITECONF/local/JobConfig/site-local-config.xml \
    -b "STABLE BEAMS" --normtag=/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json \
    -u /pb --byls --output-style csv -i /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/ReReco/Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON_v1.txt > lumi2017.csv

brilcalc lumi -c /cvmfs/cms.cern.ch/SITECONF/local/JobConfig/site-local-config.xml \
    -b "STABLE BEAMS" --normtag=/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json \
    -u /pb --byls --output-style csv -i /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/ReReco/Cert_314472-325175_13TeV_17SeptEarlyReReco2018ABC_PromptEraD_Collisions18_JSON.txt > lumi2018.csv


~~~
--->