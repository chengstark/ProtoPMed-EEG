#!/bin/bash
#SBATCH --gres=gpu:2080rtx:1 --constraint=[p100|2080rtx] -p compsci-gpu --time=10-00:00:00
#SBATCH --mem=128G

echo "start running"
nvidia-smi

model_path="/usr/xtmp/zg78/proto_proj/saved_models/eeg_model/07-25-2022_17-11_random=44_class_weighted_m_OFF_identity_CVall_fullset_1658783472/40_2push0.6683.pth"
train_dir="/usr/xtmp/zg78/proto_proj/data/10_train_test_split_50s_/train/"
test_dir="/usr/xtmp/zg78/proto_proj/data/10_train_test_split_50s_/test/"


python ../helpers/global_analysis_annoy.py  \
    -model_path $model_path \
    -train_dir $train_dir \
    -test_dir $test_dir
