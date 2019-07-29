#!/bin/bash

set -e
export PYTHONPATH=hepaccelerate:coffea:.
export HEPACCELERATE_CUDA=0
export KERAS_BACKEND="tensorflow"

source ~/Documents/root-build/bin/thisroot.sh

python3 tests/hmm/testmatrix.py