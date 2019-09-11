import os, sys
import glob
sys.path.append('../')
sys.path.append('../hepaccelerate/')
sys.path.append('../tests/hmm/')
from pars import datasets
from hmumu_utils import create_dataset_jobfiles
import json

filenames_cache = {}
chunksize_multiplier = {}
jobfile_data = []
veto_datasets = []
datapath = "/mnt/hadoop/"
cachepath = "/depot/cms/hmm/cache/"
missing = 0
total = 0

for dataset in datasets:
    dataset_name, dataset_era, dataset_globpattern, is_mc = dataset
    filenames_all = glob.glob(datapath + dataset_globpattern, recursive=True)
    filenames_all = [fn for fn in filenames_all if not "Friend" in fn]
    if dataset_name+"_"+dataset_era not in veto_datasets:
        filenames_cache[dataset_name + "_" + dataset_era] = [fn.replace(datapath, "") for fn in filenames_all]

        if len(filenames_all) == 0:
            raise Exception("Dataset {0} matched 0 files from glob pattern {1}, verify that the data files are located in {2}".format(dataset_name, dataset_globpattern, datapath))

        try:
            filenames_all = filenames_cache[dataset_name + "_" + dataset_era]
        except KeyError as e:
            print("Could not load {0}, please make sure this dataset has been added to cache".format(dataset_name + "_" + dataset_era))
            raise e

    filenames_all_full = []
    for f in filenames_all:
        total += 1
        if not glob.glob(cachepath+f.replace(".root", "*")):
            filenames_all_full.append(datapath + f)
            missing += 1
#    chunksize = chunksize_multiplier.get(dataset_name, 1)
#    print("Saving dataset {0}_{1} with {2} files in {3} files per chunk to jobfiles".format(dataset_name, dataset_era, len(filenames_all_full), chunksize))
#    jobfile_dataset = create_dataset_jobfiles(dataset_name, dataset_era,filenames_all_full, is_mc, chunksize, "./")
#    jobfile_data += jobfile_dataset
#    print("Dataset {0}_{1} consists of {2} chunks".format(dataset_name, dataset_era, len(jobfile_dataset)))

print("{0}:  Missing cache for {1} files out of {2}".format(cachepath, missing, total))
