#!/bin/bash
#SBATCH --mem=256g --time=10-00:00:00
#SBATCH --cpus-per-task=24

echo "start running"

source /home/users/zg78/miniconda3/bin/activate
conda activate proto

python ../src/dataset.py
