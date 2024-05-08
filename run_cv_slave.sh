#!/bin/bash
#SBATCH  --gres=gpu:2080rtx:2 --constraint=[p100|2080rtx] -p compsci-gpu --time=10-00:00:00
#SBATCH --mem=128G

echo "start running"
nvidia-smi

echo "fold_idx $1"

python main.py  -experiment_run="CV_miniset_test_fold_${1}" \
    -last_layer_weight=-1 \
    -random_seed=44 \
    -gpuid=0,1 \
    -latent_space_type='arc' \
    -add_on_layers_type='identity' \
    -model_dir_root='/usr/xtmp/zg78/proto_proj/saved_models' \
    -m=0 \
    -CV_fold=$1