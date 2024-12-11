#!/bin/bash
#SBATCH  --gres=gpu:2080rtx:1 --constraint=[p100|2080rtx] -p compsci-gpu --time=10-00:00:00

echo "start running"
nvidia-smi

srun -u python ../helpers/local_analysis_v2.py  -sample_root_dir '/usr/xtmp/zg78/proto_proj/data/10_train_test_split_50s_/' \
                                -model_path '/usr/xtmp/zg78/proto_proj/saved_models/eeg_model/06-06-2022_13-44_random=44_class_weighted_m_0.05_width_5_1654537487/10push0.6220.pth'
