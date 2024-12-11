import os
import torch
import torch.utils.data
import numpy as np
import pickle as pkl
import pandas as pd
from tqdm import tqdm
import argparse
import torch.nn.functional as F
from dataHelper import EEGDataset
from helpers import json_load


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-sample_root_dir', type=str, required=True, help="Root directory for samples.")
    parser.add_argument('-model_path', type=str, required=True, help="Path to the trained model.")
    return parser.parse_args()


def setup_directories(sample_root_dir, model_path):
    """
    Set up directories for data and saving results.

    Args:
        sample_root_dir (str): Root directory for samples.
        model_path (str): Path to the model.

    Returns:
        dict: Dictionary containing paths for test, train, and save directories.
    """
    test_samples_dir = os.path.join(sample_root_dir, 'test/')
    train_samples_dir = os.path.join(sample_root_dir, 'train/')
    save_dir = os.path.join(model_path[:-4], f"_extracted_features/{test_samples_dir.replace('/', '_')}/")
    os.makedirs(save_dir, exist_ok=True)

    return {
        'test_samples_dir': test_samples_dir,
        'train_samples_dir': train_samples_dir,
        'save_dir': save_dir,
        'protos_dir': '/'.join(model_path.split('/')[:-1]) + '/protos/'
    }


def load_model(model_path):
    """
    Load the trained model.

    Args:
        model_path (str): Path to the trained model.

    Returns:
        tuple: Loaded model and its DataParallel version.
    """
    print(f'Loading model from {model_path}', flush=True)
    ppnet = torch.load(model_path).cuda()
    return ppnet, torch.nn.DataParallel(ppnet)


def sanity_check(ppnet, train_samples_dir, protos_dir):
    """
    Perform sanity checks on the prototypes.

    Args:
        ppnet (torch.nn.Module): Loaded model.
        train_samples_dir (str): Directory for training samples.
        protos_dir (str): Directory for prototypes.
    """
    train_vote_dict = pkl.load(open(os.path.join(train_samples_dir, 'votes_dict.pkl'), 'rb'))
    prototype_info = json_load(os.path.join(
        protos_dir, f"epoch-{model_path.split('/')[-1].split('push')[0].split('_')[0]}",
        'global_max_proto_filenames_dict.json'))

    prototype_sample_identity = [train_vote_dict[v[:-4]] for v in prototype_info.values()]

    print(f'Prototypes are chosen from {len(set(prototype_sample_identity))} classes.', flush=True)
    print(f'Their class identities are: {prototype_sample_identity}', flush=True)

    prototype_max_connection = torch.argmax(ppnet.last_layer.weight, dim=0).cpu().numpy()
    if np.all(prototype_max_connection == prototype_sample_identity):
        print('All prototypes connect most strongly to their respective classes.', flush=True)
    else:
        print('WARNING: Not all prototypes connect most strongly to their respective classes.', flush=True)


def analyze_samples(ppnet_multi, test_img_loader, k, save_dir):
    """
    Analyze the test samples and extract top-class information.

    Args:
        ppnet_multi (torch.nn.DataParallel): DataParallel model.
        test_img_loader (DataLoader): DataLoader for test samples.
        k (int): Number of top classes to consider.
        save_dir (str): Directory to save results.
    """
    top_class_info_dict = {
        'sample_id': [],
        'similarity score vector': []
    }

    for i in range(k):
        top_class_info_dict[f'top {i+1} overall_proto index'] = []
        top_class_info_dict[f'top {i+1} overall_proto class similarity score'] = []
        top_class_info_dict[f'top {i+1} overall_proto class connection strength'] = []
        top_class_info_dict[f'top {i+1} overall_proto positive class contribution for top class'] = []
        top_class_info_dict[f'top {i+1} class proto index'] = []
        top_class_info_dict[f'top {i+1} class similarity score'] = []
        top_class_info_dict[f'top {i+1} class connection strength'] = []

    for signal, vote, sample_id in tqdm(test_img_loader):
        top_class_info_dict['sample_id'].append(sample_id[0][:-4])
        images_test = signal.cuda()
        _, logits, prototype_activations = ppnet_multi(images_test)

        top_class_info_dict['similarity score vector'].append(prototype_activations[0].detach().cpu().numpy())
        topk_logits, topk_classes = torch.topk(logits[0], k=k)

        for i, j in enumerate(torch.topk(prototype_activations[0], k=k).indices.cpu().numpy()):
            top_class_info_dict[f'top {i+1} overall_proto index'].append(j)
            top_class_info_dict[f'top {i+1} overall_proto class similarity score'].append(
                prototype_activations[0][j].item())
            top_class_info_dict[f'top {i+1} overall_proto class connection strength'].append(
                torch.max(ppnet.last_layer.weight[:, j]).item())

    pd.DataFrame(top_class_info_dict).to_csv(f'{save_dir}/top_class_info.csv', index=False)


def main():
    args = parse_args()
    dirs = setup_directories(args.sample_root_dir, args.model_path)

    ppnet, ppnet_multi = load_model(args.model_path)
    sanity_check(ppnet, dirs['train_samples_dir'], dirs['protos_dir'])

    test_img_dataset = EEGDataset(dirs['test_samples_dir'])
    test_img_loader = torch.utils.data.DataLoader(
        test_img_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)

    analyze_samples(ppnet_multi, test_img_loader, k=3, save_dir=dirs['save_dir'])


if __name__ == "__main__":
    main()
