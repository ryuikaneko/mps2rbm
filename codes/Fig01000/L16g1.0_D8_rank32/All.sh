#!/bin/bash

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate ml

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python main.py

conda deactivate
