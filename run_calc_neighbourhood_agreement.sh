#!/bin/bash
#SBATCH --mem=128G


# json_dir="/usr/xtmp/zg78/proto_proj/saved_models/eeg_model/07-25-2022_17-11_random=44_class_weighted_m_OFF_identity_CVall_fullset_1658783472/40_2push0.6683_global_analysis_annoy/"
json_dir="/usr/xtmp/zg78/proto_proj/interp_eeg/blackbox_model_neighbourhodd_analysis/"
train_dir="/usr/xtmp/zg78/proto_proj/data/10_train_test_split_50s_rerun/train/"
test_dir="/usr/xtmp/zg78/proto_proj/data/10_train_test_split_50s_rerun/test/"


python calc_neighbourhood_agreement.py  \
    -json_dir $json_dir \
    -train_dir $train_dir \
    -test_dir $test_dir
