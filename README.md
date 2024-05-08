This code package implements ProtoPMed-EEG from the paper "Mapping the Ictal-Interictal-Continuum 
Using Interpretable Machine Learning (ProtoPMed-EEG)" by Alina Jade Barnett* (Duke University), 
Zhicheng Guo* (Duke University), Wendong Ge (Harvard University), Brandon Westover (Harvard University), 
Jin Jing (Harvard University) and Cynthia Rudin (Duke University) (* denotes equal contribution).

This code package was SOLELY developed by the authors at Duke University and Harvard University.

Prerequisites: PyTorch version 1.10.2 
Recommended hardware: 1 to 2 NVIDIA Tesla P-100 GPUs, or 2 NVIDIA Tesla K-80 GPUs, or 1 to 2 NVIDIA 2080RTX GPUs

Note: the .sh files refernced here have been set up to work with a slurm batch submission system. 
      If you do not have a slurm batch submission system, you can run them by typing "source 
      filename.sh" in the command line.

Instructions for preparing the data:
1. The data processing code exists in dataHelper.py
2. Use the function multiprocess_preprocess_signals three times to create the directories that will 
   become the training, test and push directories.

Instructions for running the model on a test set:
1. Run run_generate_csv.sh. This runs local_analysis_v2.py and generate_csv.py.

Instructions for finding the nearest prototypes to a test sample:
1. Look at the run_generate_csv.sh. 

Instructions for training the model:
1. In settings.py, provide the appropriate strings for data_path, train_dir, test_dir,
train_push_dir:
(1) data_path is where the datasets reside
(2) train_dir is the directory containing the training set
(3) test_dir is the directory containing the test set
(4) train_push_dir is the directory of EEG samples that are eligible to become prototypes
2. Run run.sh

Instructions for finding the nearest samples to each prototype:
1. Run run_global_analysis_annoy.sh