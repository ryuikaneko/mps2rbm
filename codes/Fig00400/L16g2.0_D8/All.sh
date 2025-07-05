#!/bin/bash

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate ml

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

for rank in \
8 16 24 32 40 48 56 64
do

for seed in \
`seq 12340 12349`
do

python main.py --seed ${seed} --rank ${rank} --g 2.0

done

done

conda deactivate
