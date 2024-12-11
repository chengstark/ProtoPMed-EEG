import torch
import torch.utils.data
import json
import os
from dataHelper import EEGDatasetCustom
from annoy import AnnoyIndex
import argparse
from tqdm import tqdm
import numpy as np

K = 10

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpuid', type=str, default='0')
    parser.add_argument('-model_path', type=str)
    parser.add_argument('-test_dir', type=str)
    parser.add_argument('-train_dir', type=str)
    parser.add_argument('-width', type=int, default=5)
    return parser.parse_args()

def find_k_neighbours_annoy(k, model, dataloader, fns, split, width, save_dir):
    """
    Find k-nearest neighbors using Annoy.

    Args:
        k (int): Number of neighbors.
        model (nn.Module): Model to extract features.
        dataloader (DataLoader): Dataloader for the dataset.
        fns (list): List of file names.
        split (str): Split type (train/test).
        width (int): Width multiplier for feature size.
        save_dir (str): Directory to save Annoy indices and neighbors.
    """
    all_features = None

    for _, (signal, _, _) in enumerate(tqdm(dataloader)):
        signal = signal.cuda()
        feature = model.conv_features(signal).squeeze().detach().cpu().numpy()
        if all_features is None:
            all_features = feature
        else:
            all_features = np.concatenate((all_features, feature), axis=0)

    print('Building Annoy index', flush=True)
    annoy_index = AnnoyIndex(all_features.shape[1], 'angular')
    annoy_index.set_seed(1)
    for idx, feat in enumerate(tqdm(all_features)):
        annoy_index.add_item(idx, feat)
    annoy_index.build(1000, n_jobs=-1)
    annoy_index.save(f'{save_dir}/annoy_index_{split}.ann')
    print(f'Annoy index saved to {save_dir}/annoy_index_{split}.ann', flush=True)

    neighbours_dict = {}
    for idx, feat in enumerate(tqdm(all_features)):
        neighbours = annoy_index.get_nns_by_item(idx, n=k+1, search_k=-1, include_distances=False)
        neighbours_dict[fns[idx]] = [fns[i] for i in neighbours[1:]]

    with open(f'{save_dir}/{split}_neighbour_{k}_sample_id_dict.json', 'w') as f:
        json.dump(neighbours_dict, f)
    print(f'Neighbors saved to {save_dir}/{split}_neighbour_{k}_sample_id_dict.json', flush=True)

if __name__ == "__main__":
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

    model_path = args.model_path
    save_dir = model_path[:-4] + "_global_analysis_annoy/"
    os.makedirs(save_dir, exist_ok=True)

    train_dir = args.train_dir
    test_dir = args.test_dir

    train_dataset = EEGDatasetCustom(train_dir)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)
    train_fns = train_dataset.get_signal_fns()

    test_dataset = EEGDatasetCustom(test_dir)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)
    test_fns = test_dataset.get_signal_fns()

    print(f'Loading model from {model_path}', flush=True)
    ppnet = torch.load(model_path).cuda()

    find_k_neighbours_annoy(k=K, model=ppnet, dataloader=train_loader, fns=train_fns, split='train', width=args.width, save_dir=save_dir)
    find_k_neighbours_annoy(k=K, model=ppnet, dataloader=test_loader, fns=test_fns, split='test', width=args.width, save_dir=save_dir)
