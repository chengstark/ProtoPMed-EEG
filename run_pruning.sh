#!/bin/bash
#SBATCH  --gres=gpu:1 --constraint=[v100|p100] -p compsci-gpu --job-name=prune

source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"
nvidia-smi

srun python pruning.py -modeldir='/usr/xtmp/mammo/alina_saved_models/vgg16/0125_topkk=9_fa=0.001_random=4/' \
                            -model='50_9push0.9645.pth' \
                            -train_dir="/usr/xtmp/mammo/Lo1136i_resplit_with_fa_3class/train_plus_val_augmented/" \
                            -push_dir="/usr/xtmp/mammo/Lo1136i_resplit_finer/by_margin/train_plus_val/" \
                            -test_dir="/usr/xtmp/mammo/Lo1136i_resplit_3class/test_DONOTTOUCH/" 

#"/usr/xtmp/mammo/Lo1136i_resplit_with_fa_3class/train_plus_val/" \ /usr/xtmp/mammo/Lo1136i_resplit_finer/by_margin/train_plus_val/
