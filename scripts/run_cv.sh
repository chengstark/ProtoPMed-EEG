echo "start running cv"

# for spec_coef in 0 0.1 1 2 5 10
# do
#     echo $spec_coef
#     for fold_idx in {0..4}
#     do
#         echo $fold_idx
#         sbatch run_cv_slave.sh $fold_idx $spec_coef
#     done;
# done;


for fold_idx in {0..4}
do
    echo $fold_idx
    sbatch run_cv_slave.sh $fold_idx
done;
