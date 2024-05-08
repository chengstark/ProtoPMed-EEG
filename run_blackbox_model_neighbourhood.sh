#!/bin/bash
#SBATCH  --gres=gpu:2080rtx:1 --constraint=[p100|2080rtx] -p compsci-gpu --time=10-00:00:00 --mem=128G

echo "start running"
nvidia-smi

python3 blackbox_model_neighbourhood.py \
        -train_dir='/usr/xtmp/zg78/proto_proj/data/10_train_test_split_50s_/train/' \
        -test_dir='/usr/xtmp/zg78/proto_proj/data/10_train_test_split_50s_/test/'
