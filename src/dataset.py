from multiprocessing.sharedctypes import Value
import numpy as np
import os
import pickle as pkl
from torch.utils.data import Dataset
from scipy.io import loadmat
from mat73 import loadmat as loadmat73
from tqdm.auto import tqdm
from mne.filter import filter_data, notch_filter
import torch
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from multiprocessing import Pool, cpu_count
import shutil
import multiprocessing
import random

from sklearn.model_selection import KFold
from collections import defaultdict

class EEGDataset(Dataset):
    """
    EEG dataset for loading signals and corresponding votes.

    Args:
        root_dir (str): Directory containing signal files.
        aug (bool): Whether to apply augmentation.
    """
    def __init__(self, root_dir, aug=False):
        super(EEGDataset, self).__init__()
        self.aug = aug
        self.root_dir = root_dir
        self.signals = []
        self.signal_fns = []

        print(f'[Dataloader] Loading data from {root_dir}', flush=True)
        files = sorted(os.listdir(self.root_dir))

        for idx, fn in enumerate(files):
            if fn.endswith('.npy'):
                self.signals.append(np.load(f'{self.root_dir}/{fn}'))
                self.signal_fns.append(fn)

                if idx % 2000 == 0:
                    print(f'[Dataloader] Loading data {idx}', flush=True)

        print(f'[Dataloader] Loading data finished', flush=True)
        self.votes_dict = pkl.load(open(f'{self.root_dir}/votes_dict.pkl', 'rb'))

    def __len__(self):
        return len(self.signal_fns)

    def __getitem__(self, idx):
        sample_id = self.signal_fns[idx]
        signal = torch.from_numpy(self.signals[idx]).type(torch.FloatTensor)
        if self.aug:
            if torch.rand(1) < 0.5:
                signal[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]] = signal[[4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11]]

        vote = torch.from_numpy(np.asarray(self.votes_dict[self.signal_fns[idx][:-4]])).type(torch.long)
        return signal, vote, sample_id

class EEGDataset_CV(Dataset):
    """
    EEG dataset for cross-validation.

    Args:
        root_dir (str): Directory containing signal files.
        n_folds (int): Number of cross-validation folds.
        aug (bool): Whether to apply augmentation.
    """
    def __init__(self, root_dir, n_folds=5, aug=False):
        super(EEGDataset_CV, self).__init__()
        self.aug = aug
        self.root_dir = root_dir
        self.split = self.root_dir.split('/')[-2]
        self.signals = []
        self.signal_fns = []
        self.patient_list = []
        self.fold_sample_dict = defaultdict(lambda: [[], []])
        self.length = len(self.signal_fns)

        print(f'[Dataloader] Loading data from {root_dir}', flush=True)
        files_list = os.listdir(self.root_dir)
        for idx, fn in enumerate(sorted(files_list)):
            if fn.endswith('.npy'):
                self.signals.append(np.load(f'{self.root_dir}/{fn}'))
                self.signal_fns.append(fn)

                if idx % 500 == 0:
                    print(f'[Dataloader] Loading data {idx}', flush=True)

        print(f'[Dataloader] Loading data finished', flush=True)
        self.votes_dict = pkl.load(open(f'{self.root_dir}/votes_dict.pkl', 'rb'))
        self.construct_folds(n_folds)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        idx = self.sample_idxs[index]
        sample_id = self.signal_fns[idx]
        signal = torch.from_numpy(self.signals[idx]).type(torch.FloatTensor)
        if self.aug:
            if torch.rand(1) < 0.5:
                signal[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]] = signal[[4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11]]

        vote = torch.from_numpy(np.asarray(self.votes_dict[self.signal_fns[idx][:-4]])).type(torch.long)
        return signal, vote, sample_id

    def construct_folds(self, n_folds):
        """
        Construct folds for cross-validation.

        Args:
            n_folds (int): Number of folds.
        """
        self.patient_list = list(set([fn.split('_')[0] for fn in self.signal_fns]))
        kf = KFold(n_splits=n_folds)

        fold_index = 0
        for train_index, val_index in kf.split(self.patient_list):
            self.fold_sample_dict[fold_index][0] = [index for index, fn in enumerate(self.signal_fns) if fn.split('_')[0] in np.asarray(self.patient_list)[train_index]]
            self.fold_sample_dict[fold_index][1] = [index for index, fn in enumerate(self.signal_fns) if fn.split('_')[0] in np.asarray(self.patient_list)[val_index]]
            fold_index += 1

    def set_train(self, fold_index):
        """
        Set dataset to training samples for a specific fold.

        Args:
            fold_index (int): Fold index.
        """
        self.sample_idxs = self.fold_sample_dict[fold_index][0]
        self.length = len(self.sample_idxs)

    def set_val(self, fold_index):
        """
        Set dataset to validation samples for a specific fold.

        Args:
            fold_index (int): Fold index.
        """
        self.sample_idxs = self.fold_sample_dict[fold_index][1]
        self.length = len(self.sample_idxs)

    def set_all(self):
        """
        Use all samples in the dataset.
        """
        self.sample_idxs = np.asarray(list(range(len(self.signal_fns))))
        self.length = len(self.sample_idxs)

class EEGProtoDataset(Dataset):
    """
    EEG dataset for prototype signals.

    Args:
        root_dir (str): Directory containing prototype signal files.
        proto_ids (list): List of prototype IDs.
    """
    def __init__(self, root_dir, proto_ids):
        super(EEGProtoDataset, self).__init__()
        self.root_dir = root_dir
        self.proto_ids = proto_ids

    def __len__(self):
        return len(self.proto_ids)

    def __getitem__(self, idx):
        proto_id = self.proto_ids[idx]
        signal = torch.from_numpy(np.load(f'{self.root_dir}/{proto_id}.npy')).type(torch.FloatTensor)
        return signal, proto_id

def save_signal_visualization(data_dir, filename, save_path, figsize=(10, 5)):
    """This function saves the visualization of a signal.
    
    Args:
        data_dir (str): The directory where the data file is located.
        filename (str): The name of the data file.
        save_path (str): The path where the visualization will be saved.
        figsize (tuple, optional): The size of the figure. Defaults to (10, 5).
    
    Raises:
        FileNotFoundError: If the data file does not exist.
    
    Returns:
        None
    """
    
    if not save_path.endswith('.jpg'):
        save_path += '.jpg'
    signal = np.load(f'{data_dir}/{filename}')
    fig = plt.figure(figsize=figsize)
    plt.plot(signal)
    plt.title(filename)
    plt.savefig(save_path)
    plt.close('all')


def preprocess_signals(src_dir, dst_dir, fn_lists, split='train'):
    """This function preprocesses signals from a source directory and saves the processed signals in a destination directory. 
    
    Args:
        src_dir (str): The source directory where the original signals are stored.
        dst_dir (str): The destination directory where the processed signals will be saved.
        fn_lists (list): A list of file names to be processed.
        split (str, optional): The type of data split. Defaults to 'train'.
    
    The function first checks if the destination directory and the split subdirectory exist, and creates them if they don't. 
    It then loads each file from the source directory, preprocesses the signal data, and saves the processed signal in the destination directory. 
    The function also creates two dictionaries, votes_dict and votes_dict_raw, which store the maximum vote and the raw votes for each file, respectively. 
    These dictionaries are saved in the split subdirectory of the destination directory.
    
    Note:
        The signal preprocessing includes the following steps:
        - Swapping certain channels, this is acoording to our medical specialist
        - Applying a notch filter
        - Applying a bandpass filter
        - Clipping the signal values to the range [-500, 500]
    """
    
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    
    if not os.path.exists(os.path.join(dst_dir, split)):
        os.mkdir(os.path.join(dst_dir, split))
    
    votes_dict = dict()
    votes_dict_raw = dict()
    for fn in tqdm(fn_lists):
        mat = loadmat(f'{src_dir}/{fn}')
        signal = np.asarray(mat['data_50sec']).astype(np.float)
        votes = np.asarray(mat['votes']).squeeze()

        # Augmentation:
        #   Channels #01 to 04 (LL) swap with Channels #05 to 08 (RL)
        #   Channels #09 to 12 (LP) swap with Channels #13 to 16 (RP)
        signal = signal[[0,4,5,6, 11,15,16,17, 0,1,2,3, 11,12,13,14]] - signal[[4,5,6,7, 15,16,17,18, 1,2,3,7, 12,13,14,18]]
        signal = notch_filter(signal, 200, 60, n_jobs=-1, verbose='ERROR')
        signal = filter_data(signal, 200, 0.5, 40, n_jobs=-1, verbose='ERROR')
        signal = np.where(signal<=500, signal, 500)
        signal = np.where(signal>=-500, signal, -500)

        votes_dict[fn] = np.argmax(votes)
        votes_dict_raw[fn] = votes
        np.save(f'{os.path.join(dst_dir, split)}/{fn}.npy', signal)

    pkl.dump(votes_dict, open(f'{os.path.join(dst_dir, split)}/votes_dict.pkl', 'wb'))
    pkl.dump(votes_dict_raw, open(f'{os.path.join(dst_dir, split)}/votes_dict_raw.pkl', 'wb'))


class EEGDatasetCustom(Dataset):
    def __init__(self, root_dir, aug=False):
        """
        Custom EEG Dataset.

        Args:
            root_dir (str): Directory containing EEG data.
            aug (bool): Whether to apply data augmentation.
        """
        super(EEGDatasetCustom).__init__()

        self.aug = aug
        self.root_dir = root_dir
        self.signals = []
        self.signal_fns = []

        print(f'[Dataloader] Loading data from {root_dir}', flush=True)

        files = sorted(os.listdir(self.root_dir))
        print(f'[Dataloader] {len(files)} files to be loaded', flush=True)

        for idx, fn in enumerate(files):
            if fn.endswith('.npy'):
                self.signals.append(np.load(f'{self.root_dir}/{fn}'))
                self.signal_fns.append(fn)

                if idx % 2000 == 0:
                    print(f'[Dataloader] Loading data {idx}', flush=True)

        print(f'[Dataloader] Loading data finished', flush=True)

        self.votes_dict = pkl.load(open(f'{self.root_dir}/votes_dict.pkl', 'rb'))

    def __len__(self):
        return len(self.signal_fns)

    def __getitem__(self, idx):
        sample_id = self.signal_fns[idx]
        signal = torch.from_numpy(self.signals[idx]).type(torch.FloatTensor)
        if self.aug:
            if torch.rand(1) < 0.5:
                signal[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]] = signal[[4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11]]

        vote = torch.from_numpy(np.asarray(self.votes_dict[self.signal_fns[idx][:-4]])).type(torch.long)
        return signal, vote, sample_id

    def get_signal_fns(self):
        return self.signal_fns
    

def preprocess_signal_child(fn, src_dir, dst_dir, split, mat73):
    """This function preprocesses the signal data for a child. It loads the data from a .mat file, applies a notch filter and a bandpass filter, clips the signal to a range of -500 to 500, and saves the processed signal as a .npy file.
    
    Args:
        fn (str): The filename of the .mat file to be processed.
        src_dir (str): The source directory where the .mat file is located.
        dst_dir (str): The destination directory where the processed .npy file will be saved.
        split (str): The split of the data (e.g., 'train', 'test').
        mat73 (bool): A flag indicating whether the .mat file is in the v7.3 format.
    
    Returns:
        tuple: A tuple containing the filename and the votes associated with the signal.
    
    Raises:
        FileNotFoundError: If the .mat file cannot be found in the source directory.
        ValueError: If the signal data in the .mat file is not in the expected format.
    """
    
    if mat73:
        mat = loadmat73(f'{src_dir}/{fn}')
    else:
        mat = loadmat(f'{src_dir}/{fn}')
    signal = np.asarray(mat['data_50sec']).astype(np.float)
    votes = np.asarray(mat['votes']).squeeze()
    # 'Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz,'Fp2','F4','C4','P4','F8','T4','T6','O2', ‘EKG’
    # Augmentation:
    #   Channels #01 to 04 (LL) swap with Channels #05 to 08 (RL)
    #   Channels #09 to 12 (LP) swap with Channels #13 to 16 (RP)
    signal = signal[[0,4,5,6, 11,15,16,17, 0,1,2,3, 11,12,13,14]] - signal[[4,5,6,7, 15,16,17,18, 1,2,3,7, 12,13,14,18]]
    signal = notch_filter(signal, 200, 60, n_jobs=-1, verbose='ERROR')
    signal = filter_data(signal, 200, 0.5, 40, n_jobs=-1, verbose='ERROR')

    signal = np.where(signal<=500, signal, 500)
    signal = np.where(signal>=-500, signal, -500)

    np.save(f'{os.path.join(dst_dir, split)}/{fn}.npy', signal)

    return fn, votes


def multiprocess_preprocess_signals(src_dir, dst_dir, fn_lists, split='train', n_jobs=2, mat73=False):

    """
    Process signal data files using multiple processes and save the output in specified directories.

    This function sets up directories for storing processed data, initializes a multiprocessing
    pool to preprocess files, and saves results in pickle files. It handles .mat files, optionally
    accounting for the v7.3 format.

    Parameters:
    - src_dir (str): Source directory containing .mat files.
    - dst_dir (str): Destination directory for processed files.
    - fn_lists (list of str): List of filenames to process. If None, processes all .mat files in src_dir.
    - split (str): Subdirectory within dst_dir to store processed files (default: 'train').
    - n_jobs (int): Number of worker processes to use (default: 2).
    - mat73 (bool): Set to True if files are MATLAB v7.3 format (default: False).

    The function does not return any value but outputs two pickle files: 'votes_dict.pkl' and 'votes_dict_raw.pkl',
    containing processed data results.

    Example:
    ```python
    multiprocess_preprocess_signals('/path/to/source', '/path/to/destination', ['data1.mat', 'data2.mat'])
    ```
    """

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    
    if not os.path.exists(os.path.join(dst_dir, split)):
        os.mkdir(os.path.join(dst_dir, split))
    
    votes_dict = dict()
    votes_dict_raw = dict()

    print('\tinitializing pool', flush=True)

    if fn_lists is None:
        fn_lists = [x for x in os.listdir(src_dir) if x.endswith('.mat')]

    pool_args = []
    for idx, fn in enumerate(fn_lists):
        pool_args.append([fn, src_dir, dst_dir, split, mat73])
    
    print('\trunning pool', flush=True)
    pool = Pool(n_jobs)
    rets = []
    for ret in tqdm(pool.istarmap(preprocess_signal_child, pool_args), total=len(pool_args)):
        rets.append(ret)

    pool.terminate()

    print('\tpool returns', flush=True)
    for ret in tqdm(rets):
        fn, votes = ret
        votes_dict[fn] = np.argmax(votes)
        votes_dict_raw[fn] = votes

    pkl.dump(votes_dict, open(f'{os.path.join(dst_dir, split)}/votes_dict.pkl', 'wb'))
    pkl.dump(votes_dict_raw, open(f'{os.path.join(dst_dir, split)}/votes_dict_raw.pkl', 'wb'))


def extract_spec_child(fn, src_dir, dst_dir, split):
    """This function extracts specific child elements from a given source directory and saves them into a destination directory.
    
    Args:
        fn (str): The filename of the file to be processed.
        src_dir (str): The source directory where the file is located.
        dst_dir (str): The destination directory where the processed file will be saved.
        split (str): The split type of the data.
    
    Returns:
        None. The function saves the processed data into a .npy file in the destination directory.
    """
    
    mat = loadmat(f'{src_dir}/{fn}')
    spec_meta = mat['spec_10min']
    spec_mid10s = []
    for i in range(4):
        spec_mid10s.append(spec_meta[:, 1][i][:, int(295*0.5):int(305*0.5)])

    spec_mid10s = np.asarray(spec_mid10s)
    np.save(f'{os.path.join(dst_dir, split)}/{fn}.npy', spec_mid10s)


def extract_spec(src_dir, dst_dir, fn_lists, split='train', n_jobs=2):
    """This function is used to extract specifications from a source directory and save them in a destination directory. It uses multiprocessing to speed up the process.
    
    Args:
        src_dir (str): The source directory from where the specifications are to be extracted.
        dst_dir (str): The destination directory where the extracted specifications are to be saved.
        fn_lists (list): A list of file names from which specifications are to be extracted.
        split (str, optional): The type of data split. Defaults to 'train'.
        n_jobs (int, optional): The number of jobs to run in parallel. Defaults to 2.
    
    Raises:
        OSError: If the destination directory does not exist, it will be created.
    
    Note:
        This function uses the 'Pool' class from the 'multiprocessing' module to create a pool of worker processes. It also uses the 'tqdm' module to show a progress bar.
    """
    
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    
    if not os.path.exists(os.path.join(dst_dir, split)):
        os.mkdir(os.path.join(dst_dir, split))

    print('\tinitializing pool', flush=True)
    pool_args = []
    for idx, fn in enumerate(fn_lists):
        pool_args.append([fn, src_dir, dst_dir, split])
    
    print('\trunning pool', flush=True)
    pool = Pool(n_jobs)
    for _ in tqdm(pool.istarmap(extract_spec_child, pool_args), total=len(pool_args)): pass
    pool.terminate()


def votes_leq_20(votes):
    """This function checks if the total sum of votes is greater than or equal to 20.
    
    Args:
        votes (list): A list of integers representing votes.
    
    Returns:
        bool: True if the sum of votes is greater than or equal to 20, False otherwise.
    """

    return sum(votes) >= 20

def votes_unanimous_85pct(votes):
    """This function checks if the maximum vote count is at least 85% of the total vote count.
    
    Args:
        votes (list): A list of integers representing vote counts.
    
    Returns:
        bool: True if the maximum vote count is at least 85% of the total vote count, False otherwise.
    """
    
    return (max(votes) / sum(votes)) >= 0.85

def create_push_folder(dataset_dir, condition_function):
    """This function creates a new directory for a specific condition within a dataset directory and copies relevant files into it. 
    
    Args:
        dataset_dir (str): The directory path where the dataset is located.
        condition_function (function): The function that defines the condition for which the new directory is created.
    
    The function first checks if a directory for the condition already exists within the dataset directory. If not, it creates one. 
    It then loads two dictionaries from pickle files: 'votes_dict_raw.pkl' and 'votes_dict.pkl'. 
    It creates a list of sample IDs that meet the condition defined by 'condition_function' and copies the corresponding '.npy' files into the new directory. 
    It also creates two new dictionaries that only include the entries for the sample IDs that meet the condition and saves them as pickle files in the new directory.
    
    Note: The function assumes that the 'votes_dict_raw.pkl' and 'votes_dict.pkl' files exist in a 'train' subdirectory within the dataset directory.
    """

    condition_name = condition_function.__name__
    if not os.path.exists(f'{dataset_dir}/push_{condition_name}/'):
        os.mkdir(f'{dataset_dir}/push_{condition_name}/')

    train_votes_dict_raw = pkl.load(open(f'{dataset_dir}/train/votes_dict_raw.pkl', 'rb'))
    train_votes_dict = pkl.load(open(f'{dataset_dir}/train/votes_dict.pkl', 'rb'))

    proto_pool = []
    proto_pool_votes_dict = dict()
    proto_pool_votes_dict_raw = dict()
    for sample_id, votes in tqdm(train_votes_dict_raw.items()):

        if condition_function(votes):
            proto_pool.append(sample_id)

    for sample_id in tqdm(proto_pool):
        shutil.copyfile(f'{dataset_dir}/train/{sample_id}.npy', f'{dataset_dir}/push_{condition_name}/{sample_id}.npy')
        proto_pool_votes_dict[sample_id] = train_votes_dict[sample_id]
        proto_pool_votes_dict_raw[sample_id] = train_votes_dict_raw[sample_id]

    pkl.dump(proto_pool_votes_dict, open(f'{dataset_dir}/push_{condition_name}/votes_dict.pkl', 'wb'))
    pkl.dump(proto_pool_votes_dict_raw, open(f'{dataset_dir}/push_{condition_name}/votes_dict_raw.pkl', 'wb'))
    

if __name__ == '__main__':
    number_of_cores = multiprocessing.cpu_count()
    print(number_of_cores)
    print('processing external_test', flush=True)
    multiprocess_preprocess_signals('/usr/xtmp/zg78/proto_proj/data/external_test/', '/usr/xtmp/zg78/proto_proj/data/external_test/', None, split='test', n_jobs=number_of_cores, mat73=True)

