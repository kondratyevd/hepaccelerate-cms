How to submit analysis jobs:
===

The input and output directories are defined in `hmm_analyze.py`.

The submission script `submit_analyze.sh` can be ran with two arguments, for example `. submit_analyze.sh cms 20`. 
The first argument is the name of PBS queue ("cms" by default), and the second argument is number of chunks per job (20 by default). 

Full analysis (2016+17+18) contains about 4000 chunks, so by default ~200 jobs will be created.


~~
cd batch_purdue_pbs

# Set up VOMS proxy
. setup_proxy.sh

# Submit PBS jobs
. submit_analyze.sh
# wait for completion.. 

# Merge results and produce plots
. post.sh

~~

