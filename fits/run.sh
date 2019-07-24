#!/bin/bash

#dcard="hist__dimuon__inv_mass"
#dcard="hist__dimuon_invmass_70_110_cat5__dijet_inv_mass"
#dcard="hist__dimuon_invmass_70_110_cat5__leading_jet_pt"
#dcard="hist__dimuon_invmass_70_110_cat5__inv_mass"
#dcard="hist__dimuon_invmass_70_110_cat5__numt_jets"
dcard="hist__dimuon_invmass_70_110_cat5__num_soft_jets"

inp=out2
wd=`pwd`

function fit() {
    cd ../$inp/$1/datacards/2018/
    combine -D data -n $2 -M FitDiagnostics -t 0 --saveShapes --saveWithUncertainties $2.txt
    cd $wd
}

fit baseline $dcard
fit redo_jec_V16 $dcard
