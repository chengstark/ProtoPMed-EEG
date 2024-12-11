#!/bin/bash
#SBATCH  --gres=gpu:2080rtx:1 --constraint=[p100|2080rtx] -p compsci-gpu --time=10-00:00:00 --mem=128G

echo "start running"
nvidia-smi

python3 ../helpers/global_analysis.py -model_dir='/usr/xtmp/zg78/proto_proj/saved_models/eeg_model/10-05-2022_11-31_code_cleanup_check_1664983881/' \
                                   -model='10push0.1760.pth' \
                                   -train_dir='/usr/xtmp/zg78/proto_proj/data/10_train_test_split_50s_/train/' \
                                   -push_dir='/usr/xtmp/zg78/proto_proj/data/10_train_test_split_50s_/push_votes_leq_20/' \
                                   -test_dir='/usr/xtmp/zg78/proto_proj/data/10_train_test_split_50s_/test/'
