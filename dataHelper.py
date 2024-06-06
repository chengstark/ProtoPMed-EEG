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
import istarmap
import time
import shutil
import multiprocessing
import random

from sklearn.model_selection import KFold
from collections import defaultdict


class EEGDataset(Dataset):
    def __init__(self, root_dir, aug=False):
        super(EEGDataset).__init__()

        self.aug = aug
        self.root_dir = root_dir
        self.signals = []
        self.signal_fns = []

        print(f'[Dataloader] loading data from {root_dir}', flush=True)

        files = sorted(os.listdir(self.root_dir))
        # files = random.sample(list(os.listdir(self.root_dir)), 2000)
        for idx, fn in enumerate(files):
            if fn.endswith('.npy'):
                self.signals.append(np.load(f'{self.root_dir}/{fn}'))
                self.signal_fns.append(fn)

                if idx % 2000 == 0:
                    print(f'[Dataloader] loading data {idx}', flush=True)

        print(f'[Dataloader] loading data Finished', flush=True)

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
    def __init__(self, root_dir, n_folds=5, aug=False):
        super(EEGDataset).__init__()

        self.aug = aug
        self.root_dir = root_dir
        self.split = self.root_dir.split('/')[-2]
        self.signals = []
        self.signal_fns = []
        self.patient_list = []
        self.fold_sample_dict = defaultdict(lambda: [[],[]])
        self.length = len(self.signal_fns)

        print(f'[Dataloader] loading data from {root_dir}', flush=True)

        files_list = os.listdir(self.root_dir)
        # files_list = random.sample(list(os.listdir(self.root_dir)), 20)
        for idx, fn in enumerate(sorted(files_list)):
            if fn.endswith('.npy'):
                self.signals.append(np.load(f'{self.root_dir}/{fn}'))
                self.signal_fns.append(fn)

                if idx % 500 == 0:
                    print(f'[Dataloader] loading data {idx}', flush=True)

        print(f'[Dataloader] loading data Finished', flush=True)
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
        self.patient_list = list(set([fn.split('_')[0] for fn in self.signal_fns]))
        kf = KFold(n_splits=n_folds)
        
        fold_index = 0

        for train_index, val_index in kf.split(self.patient_list):
            self.fold_sample_dict[fold_index][0] = [index for index, fn in enumerate(self.signal_fns) if fn.split('_')[0] in np.asarray(self.patient_list)[train_index]]
            self.fold_sample_dict[fold_index][1] = [index for index, fn in enumerate(self.signal_fns) if fn.split('_')[0] in np.asarray(self.patient_list)[val_index]]
            fold_index += 1

    def set_train(self, fold_index):
        self.sample_idxs = self.fold_sample_dict[fold_index][0]
        self.length = len(self.sample_idxs)

    def set_val(self, fold_index):
        self.sample_idxs = self.fold_sample_dict[fold_index][1]
        self.length = len(self.sample_idxs)

    def set_all(self):
        self.sample_idxs = np.asarray(list(range(len(self.signal_fns))))
        self.length = len(self.sample_idxs)


# class EEGDataset(Dataset):
#     def __init__(self, root_dir, aug=False):
#         super(EEGDataset).__init__()

#         self.aug = aug
#         self.root_dir = root_dir
#         self.split = self.root_dir.split('/')[-2]
#         self.spec_root_dir = '/'.join(self.root_dir.split('/')[:-2]) + "_spec_mid10s/"+self.split+"/"
#         self.signals = []
#         self.specs = []
#         self.signal_fns = []

#         if not os.path.exists(self.spec_root_dir):
#             raise ValueError(f"spec_root_dir {self.spec_root_dir} doesn't exist")

#         print(f'[Dataloader] loading data from {root_dir} and {self.spec_root_dir}', flush=True)

#         for idx, fn in enumerate(sorted(os.listdir(self.root_dir))):
#             if fn.endswith('.npy'):
#                 self.signals.append(np.load(f'{self.root_dir}/{fn}'))
#                 self.signal_fns.append(fn)
#                 self.specs.append(np.load(f'{self.spec_root_dir}/{fn}'))

#                 if idx % 2000 == 0:
#                     print(f'[Dataloader] loading data {idx}', flush=True)

#         print(f'[Dataloader] loading data Finished', flush=True)
#         self.votes_dict = pkl.load(open(f'{self.root_dir}/votes_dict.pkl', 'rb'))

#     def __len__(self):
#         return len(self.signal_fns)
    
#     def __getitem__(self, idx):
#         sample_id = self.signal_fns[idx]
#         signal = torch.from_numpy(self.signals[idx]).type(torch.FloatTensor)
#         spectrogram = torch.from_numpy(self.specs[idx]).type(torch.FloatTensor)
        
#         if self.aug:
#             if torch.rand(1) < 0.5:
#                 signal[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]] = signal[[4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11]]

#         vote = torch.from_numpy(np.asarray(self.votes_dict[self.signal_fns[idx][:-4]])).type(torch.long)
#         return signal, spectrogram, vote, sample_id


class EEGProtoDataset(Dataset):
    def __init__(self, root_dir, proto_ids):
        super(EEGProtoDataset).__init__()

        self.root_dir = root_dir
        self.proto_ids = proto_ids

    def __len__(self):
        return len(self.proto_ids)
    
    def __getitem__(self, idx):
        proto_id = self.proto_ids[idx]
        signal = torch.from_numpy(np.load(f'{self.root_dir}/{proto_id}.npy')).type(torch.FloatTensor)
        
        return signal, proto_id
    

# TODO: write the real visualization function
def save_signal_visualization(data_dir, filename, save_path, figsize=(10, 5)):
    if not save_path.endswith('.jpg'):
        save_path += '.jpg'
    signal = np.load(f'{data_dir}/{filename}')
    fig = plt.figure(figsize=figsize)
    plt.plot(signal)
    plt.title(filename)
    plt.savefig(save_path)
    plt.close('all')


def preprocess_signals(src_dir, dst_dir, fn_lists, split='train'):

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
    

def preprocess_signal_child(fn, src_dir, dst_dir, split, mat73):
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
    mat = loadmat(f'{src_dir}/{fn}')
    spec_meta = mat['spec_10min']
    spec_mid10s = []
    for i in range(4):
        spec_mid10s.append(spec_meta[:, 1][i][:, int(295*0.5):int(305*0.5)])

    spec_mid10s = np.asarray(spec_mid10s)
    np.save(f'{os.path.join(dst_dir, split)}/{fn}.npy', spec_mid10s)


def extract_spec(src_dir, dst_dir, fn_lists, split='train', n_jobs=2):

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
    return sum(votes) >= 20

def votes_unanimous_85pct(votes):
    return (max(votes) / sum(votes)) >= 0.85

def create_push_folder(dataset_dir, condition_function):

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

    # multiprocessing.set_start_method('loky')

    # number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    number_of_cores = multiprocessing.cpu_count()
    print(number_of_cores)

    # train_key_3 = list(np.load('/usr/xtmp/zg78/proto_proj/data/wendong_keys/3_train_key.npy'))
    # train_key_10 = list(np.load('/usr/xtmp/zg78/proto_proj/data/wendong_keys/10_train_key.npy'))
    # test_key_10 = list(np.load('/usr/xtmp/zg78/proto_proj/data/wendong_keys/10_test_key.npy'))

    # print('processing train key 3', flush=True)
    # multiprocess_preprocess_signals('/usr/xtmp/zg78/proto_proj/data/OCT2023/', '/usr/xtmp/zg78/proto_proj/data/3_train_test_split_50s_rerun/', train_key_3, split='train', n_jobs=number_of_cores)
    # print('processing train key 10', flush=True)
    # multiprocess_preprocess_signals('/usr/xtmp/zg78/proto_proj/data/OCT2023/', '/usr/xtmp/zg78/proto_proj/data/10_train_test_split_50s_rerun/', train_key_10, split='train', n_jobs=number_of_cores)
    # print('processing test key 10', flush=True)
    # multiprocess_preprocess_signals('/usr/xtmp/zg78/proto_proj/data/OCT2023/', '/usr/xtmp/zg78/proto_proj/data/10_train_test_split_50s_rerun/', test_key_10, split='test', n_jobs=number_of_cores)

    
    print('processing external_test', flush=True)
    multiprocess_preprocess_signals('/usr/xtmp/zg78/proto_proj/data/external_test/', '/usr/xtmp/zg78/proto_proj/data/external_test/', None, split='test', n_jobs=number_of_cores, mat73=True)

    # print('processing train key 3', flush=True)
    # extract_spec('/usr/xtmp/zg78/proto_proj/data/redownloaddata/', '/usr/xtmp/zg78/proto_proj/data/3_train_test_split_spec_mid10s/', train_key_3, split='train', n_jobs=number_of_cores)
    # print('processing train key 10', flush=True)
    # extract_spec('/usr/xtmp/zg78/proto_proj/data/redownloaddata/', '/usr/xtmp/zg78/proto_proj/data/10_train_test_split_spec_mid10s/', train_key_10, split='train', n_jobs=number_of_cores)
    # print('processing test key 10', flush=True)
    # extract_spec('/usr/xtmp/zg78/proto_proj/data/redownloaddata/', '/usr/xtmp/zg78/proto_proj/data/10_train_test_split_spec_mid10s/', test_key_10, split='test', n_jobs=number_of_cores)
    # print('processing push_votes_leq_20 for 10 key', flush=True)
    # extract_spec('/usr/xtmp/zg78/proto_proj/data/redownloaddata/', '/usr/xtmp/zg78/proto_proj/data/10_train_test_split_50s__spec_mid10s/', [fn[:-4] for fn in os.listdir('/usr/xtmp/zg78/proto_proj/data/10_train_test_split_50s_/push_votes_leq_20/') if fn.endswith('.npy')], split='push_votes_leq_20', n_jobs=number_of_cores)

    # print(len(os.listdir('/usr/xtmp/zg78/proto_proj/data/combined_train_test_split/train/')))

    # vote_dict_3 = pkl.load(open('/usr/xtmp/zg78/proto_proj/data/3_train_test_split/train/votes_dict.pkl', 'rb'))
    # vote_dict_10 = pkl.load(open('/usr/xtmp/zg78/proto_proj/data/10_train_test_split/train/votes_dict.pkl', 'rb'))
    # vote_dict_combined = vote_dict_3.copy()
    # vote_dict_combined.update(vote_dict_10)

    # print(len(vote_dict_combined.keys()))
    # pkl.dump(vote_dict_combined, open('/usr/xtmp/zg78/proto_proj/data/combined_train_test_split/train/votes_dict.pkl', 'wb'))
    # print([f for f in os.listdir('/usr/xtmp/zg78/proto_proj/data/combined_train_test_split/train/') if f.endswith('.npy') and f[:-4] not in vote_dict_combined.keys()])


    # create_push_folder('/usr/xtmp/zg78/proto_proj/data/10_train_test_split_50s_rerun/', votes_leq_20)
    # create_push_folder('/usr/xtmp/zg78/proto_proj/data/10_train_test_split_50s_rerun/', votes_unanimous_85pct)


