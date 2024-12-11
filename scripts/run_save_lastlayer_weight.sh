#!/bin/bash
#SBATCH --gres=gpu:2080rtx:1 --constraint=[p100|2080rtx] -p compsci-gpu --time=10-00:00:00
#SBATCH --mem=128G

echo "start running"
nvidia-smi

# model_path="/usr/xtmp/zg78/proto_proj/saved_models/eeg_model/06-06-2022_13-44_random=44_class_weighted_m_0.05_width_5_1654537487/40push0.6427.pth"
# sample_root_dir="/usr/xtmp/zg78/proto_proj/data/10_train_test_split_50s_/"
# data_dir="${sample_root_dir}test/"
# csv_name="fullwidth_40_DistMAP"

# model_path="/usr/xtmp/zg78/proto_proj/saved_models/eeg_model/06-06-2022_14-34_random=44_class_weighted_m_0.05_width_1_proto_pool_leq20_1654540492/10push0.6236.pth"
# sample_root_dir="/usr/xtmp/zg78/proto_proj/data/10_train_test_split_50s_/"
# data_dir="${sample_root_dir}test/"
# csv_name="highvotecount_protos_DistMAP"

# model_path="/usr/xtmp/zg78/proto_proj/saved_models/eeg_model/06-06-2022_14-35_random=44_class_weighted_m_0.05_width_1_proto_pool_unan_85pct_1654540541/10push0.6117.pth"
# sample_root_dir="/usr/xtmp/zg78/proto_proj/data/10_train_test_split_50s_/"
# data_dir="${sample_root_dir}test/"
# csv_name="unan_protos_DistMAP"

model_path="/usr/xtmp/zg78/proto_proj/saved_models/eeg_model/07-25-2022_17-11_random=44_class_weighted_m_OFF_identity_CVall_fullset_1658783472/40_2push0.6683.pth"
sample_root_dir="/usr/xtmp/zg78/proto_proj/data/10_train_test_split_50s_/"
data_dir="${sample_root_dir}test/"
csv_name="base_multi_class_prototype"

echo $data_dir
echo $model_path
echo $csv_name

python ../helpers/save_lastlayer_weight.py \
    --data_dir $data_dir \
    --model_path $model_path \
    --csv_name $csv_name