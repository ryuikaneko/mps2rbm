#!/bin/bash

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate ml

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

L=40
rank=$((L/2))
seed=12345

#python main.py --seed ${seed} --rank ${rank} --g 1.0 --L ${L}
python main.py --seed ${seed} --rank ${rank} --g 0.5 --L ${L}

conda deactivate
