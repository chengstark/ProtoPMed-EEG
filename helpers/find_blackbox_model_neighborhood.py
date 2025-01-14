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

np.random.seed(1)
torch.manual_seed(1)

K = 10

class EEGDatasetCustom(Dataset):
    """
    Custom EEG Dataset for loading signals and corresponding votes.

    Args:
        root_dir (str): Directory containing EEG signal files.
        aug (bool): Whether to apply augmentation.
    """
    def __init__(self, root_dir, aug=False):
        super(EEGDatasetCustom, self).__init__()
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


def find_k_neighbours_annoy(k, model, dataloader, fns, split):
    """
    Finds k-nearest neighbors using Annoy index.

    Args:
        k (int): Number of neighbors.
        model (torch.nn.Module): Feature extraction model.
        dataloader (DataLoader): DataLoader for dataset.
        fns (list): List of sample file names.
        split (str): Split type (train/test).

    Saves:
        Annoy index and neighbors JSON.
    """
    all_features = None
    for _, (signal, _, _) in enumerate(tqdm(dataloader)):
        signal = signal.cuda()
        feature = model(signal).squeeze().detach().cpu().numpy()
        if all_features is None:
            all_features = feature
        else:
            all_features = np.concatenate((all_features, feature), axis=0)

    print('Building Annoy', flush=True)
    annoy_index = AnnoyIndex(all_features.shape[1], 'angular')
    annoy_index.set_seed(1)
    for idx, feat in enumerate(tqdm(all_features)):
        annoy_index.add_item(idx, feat)
    annoy_index.build(1000, n_jobs=-1)
    annoy_index.save(f'/usr/xtmp/zg78/proto_proj/interp_eeg/blackbox_model_neighbourhodd_analysis/annoy_index_{split}.ann')
    print('Annoy saved to', f'/usr/xtmp/zg78/proto_proj/interp_eeg/blackbox_model_neighbourhodd_analysis/annoy_index_{split}.ann', flush=True)

    neighbours_dict = {}
    for idx, feat in enumerate(tqdm(all_features)):
        neighbours = annoy_index.get_nns_by_item(idx, n=k+1, search_k=-1, include_distances=False)
        neighbours_dict[fns[idx]] = [fns[i] for i in neighbours[1:]]

    json.dump(neighbours_dict, open(f'/usr/xtmp/zg78/proto_proj/interp_eeg/blackbox_model_neighbourhodd_analysis/{split}_neighbour_{k}_sample_id_dict.json', 'w'))
    print('Saved to', f'/usr/xtmp/zg78/proto_proj/interp_eeg/blackbox_model_neighbourhodd_analysis/{split}_neighbour_{k}_sample_id_dict.json', flush=True)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpuid', nargs=1, type=str, default='0')
    parser.add_argument('-test_dir', nargs=1, type=str)
    parser.add_argument('-train_dir', nargs=1, type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

    # Load the data
    train_dir = args.train_dir[0]
    test_dir = args.test_dir[0]

    batch_size = 100

    train_dataset = EEGDatasetCustom(train_dir)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True)
    train_fns = train_dataset.get_signal_fns()

    test_dataset = EEGDatasetCustom(test_dir)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True)
    test_fns = test_dataset.get_signal_fns()

    blackbox_model = eeg_model_features(pretrained=True, width=1).cuda()
    blackbox_model_multi = torch.nn.DataParallel(blackbox_model)

    find_k_neighbours_annoy(k=K, model=blackbox_model_multi, dataloader=test_loader, fns=test_fns, split='test')


if __name__ == "__main__":
    main()
