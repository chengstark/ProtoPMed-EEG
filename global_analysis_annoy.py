import torch
import torch.utils.data
from torch.utils.data import Dataset
import json
import re
import os
from eeg_model_features import _DenseBlock, _DenseLayer, DenseNetClassifier, _Transition, DenseNetEnconder
from annoy import AnnoyIndex
import argparse
from eeg_model_features import eeg_model_features
from tqdm import tqdm
import multiprocessing
import numpy as np
import pickle as pkl


K = 10

class EEGDatasetCustom(Dataset):
    def __init__(self, root_dir, aug=False):
        super(EEGDatasetCustom).__init__()

        self.aug = aug
        self.root_dir = root_dir
        self.signals = []
        self.signal_fns = []

        print(f'[Dataloader] loading data from {root_dir}', flush=True)

        files = sorted(os.listdir(self.root_dir))
        # files = sorted(os.listdir(self.root_dir)[:200])

        print(f'[Dataloader] {len(files)} files to be loaded', flush=True)

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

    def get_signal_fns(self):
        return self.signal_fns

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-model_path', nargs=1, type=str)
parser.add_argument('-test_dir', nargs=1, type=str)
parser.add_argument('-train_dir', nargs=1, type=str)
parser.add_argument('-width', type=int, default=5)

# parser.add_argument('-dataset', nargs=1, type=str, default='cub200')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
model_path = args.model_path[0]
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

# load the data
# must use unaugmented (original) dataset
train_dir = args.train_dir[0]
test_dir = args.test_dir[0]

batch_size = 100

train_dataset = EEGDatasetCustom(train_dir)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False,
    num_workers=2, pin_memory=True)
train_fns = train_dataset.get_signal_fns()
# test set
test_dataset = EEGDatasetCustom(test_dir)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False,
    num_workers=2, pin_memory=True)
test_fns = test_dataset.get_signal_fns()

# load the model
print('load model from ' + model_path)
ppnet = torch.load(model_path)
# print('convert to maxpool logic')
# ppnet.set_topk_k(1)
ppnet = ppnet.cuda()
# ppnet_multi = torch.nn.DataParallel(ppnet)

save_dir = model_path[:-4]+f"_global_analysis_annoy/"
os.makedirs(save_dir, exist_ok=True)


def find_k_neighbours_annoy(k, model, dataloader, fns, split, wdith=5):
    
    all_features = None
    for _, (signal, _, _) in enumerate(tqdm(dataloader)):
        signal = signal.cuda()
        feature = model.conv_features(signal).squeeze().detach().cpu().numpy()
        if all_features is None:
            all_features = feature
        else:
            all_features = np.concatenate((all_features, feature), axis=0)

    print('Building Annoy', flush=True)
    annoy_index = AnnoyIndex(255*wdith, 'angular')
    annoy_index.set_seed(1)
    for idx, feat in enumerate(tqdm(all_features)):
        annoy_index.add_item(idx, feat)
    annoy_index.build(1000, n_jobs=-1)  # 10 trees
    annoy_index.save(f'{save_dir}/annoy_index_{split}.ann')
    print('Annoy saved to', f'{save_dir}/annoy_index_{split}.ann', flush=True)

    neighbours_dict = dict()

    for idx, feat in enumerate(tqdm(all_features)):
        neighbours = annoy_index.get_nns_by_item(idx, n=k+1, search_k=-1, include_distances=False)
        neighbours_dict[fns[idx]] = [fns[i] for i in neighbours[1:]]

    json.dump(neighbours_dict, open(f'{save_dir}/{split}_neighbour_{k}_sample_id_dict.json', 'w'))
    print('Saved to', f'{save_dir}/{split}_neighbour_{k}_sample_id_dict.json', flush=True)

find_k_neighbours_annoy(k=K, model=ppnet, dataloader=train_loader, fns=train_fns, split='train', wdith=args.width)
find_k_neighbours_annoy(k=K, model=ppnet, dataloader=test_loader, fns=test_fns, split='test', wdith=args.width)
