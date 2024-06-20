import gc
import numpy as np
import random
import os
import faiss
import datetime
from itertools import product
import pacmap
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from helpers import json_load
import json
import pickle as pkl
import torch
from dataHelper import EEGDataset, EEGProtoDataset
import istarmap
import pandas as pd
from scipy.io import loadmat
import multiprocessing.pool as mpp
from multiprocessing import Pool
import argparse
import torch.nn.functional as F

import multiprocessing
multiprocessing.cpu_count()


def prase_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--csv_name', type=str)
    args = parser.parse_args()

    csv_name = args.csv_name
    data_dir = args.data_dir
    data_dir_root = '/'.join(data_dir.split('/')[:-2])
    model_path = args.model_path
    save_dir = model_path[:-4]+f"_extracted_features/{data_dir.replace('/', '_')}/"
    proto_path = '/'.join(model_path.split('/')[:-1])+'/protos/epoch-'+model_path.split('/')[-1].split('push')[0].split('_')[0]+'/global_max_proto_filenames_dict.json'
    print(save_dir)
    print(proto_path)
    print(data_dir_root)

    return args, save_dir, proto_path, data_dir, data_dir_root, model_path, csv_name


def feature_extract(data_dir, model_path, save_dir):
    ppnet = torch.load(model_path)

    dataset = EEGDataset(data_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False,
        num_workers=2, pin_memory=False)

    extracted_features = []
    predictions = []
    sample_ids = []
    labels = []
    activations = []
    logits_marginlesses = []
    for _, (image, label, sample_id) in enumerate(tqdm(dataloader)):
        x = image.cuda()
        _, logits_marginless, activation = ppnet(x)
        _, preds = torch.max(logits_marginless.data, 1)
        features = ppnet.conv_features(x).squeeze().detach().cpu().numpy()
        sample_ids.append(sample_id)
        logits_marginlesses.append(logits_marginless.detach().cpu().numpy())
        predictions.append(preds.detach().cpu().numpy())
        extracted_features.append(features)
        labels.append(label)
        activations.append(activation.detach().cpu().numpy())
    logits_marginlesses = np.concatenate(logits_marginlesses, axis=0)
    extracted_features = np.concatenate(extracted_features, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    activations = np.concatenate(activations, axis=0)
    labels = np.concatenate(labels, axis=0)
    sample_ids = np.array(sample_ids, dtype="object")
    sample_ids = np.concatenate(sample_ids, axis=0)
    print(f'Extracted features with shape {extracted_features.shape}')
    
    os.makedirs(save_dir, exist_ok=True)
    np.save(f'{save_dir}/logits_marginlesses.npy', logits_marginlesses)
    np.save(f'{save_dir}/activations.npy', activations)
    np.save(f'{save_dir}/predictions.npy', predictions)
    np.save(f'{save_dir}/labels.npy', labels)
    np.save(f'{save_dir}/extracted_features.npy', extracted_features)
    np.save(f'{save_dir}/sample_ids.npy', sample_ids)


def extract_proto_probs(data_dir, proto_ids, model_path, save_dir):
    print(f'saved proto probs to {save_dir}', flush=True)
    ppnet = torch.load(model_path)

    logits_marginlesses = []
    dataset = EEGProtoDataset(data_dir, proto_ids)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False,
        num_workers=2, pin_memory=False)

    for _, (proto, proto_id) in enumerate(tqdm(dataloader)):
        x = proto.cuda()
        _, logits_marginless, _ = ppnet(x)
        logits_marginlesses.append(logits_marginless.detach().cpu().numpy())

    logits_marginlesses = np.concatenate(logits_marginlesses, axis=0)
    os.makedirs(save_dir, exist_ok=True)
    np.save(f'{save_dir}/proto_logits_marginlesses.npy', logits_marginlesses)


if __name__ == "__main__":
    print('argparse', flush=True)
    args, save_dir, proto_path, data_dir, data_dir_root, model_path, csv_name = prase_args()

    print('Feature extraction', flush=True)
    feature_extract(data_dir, model_path, save_dir)

    print('Generating prototype vectors', flush=True)
    ppnet = torch.load(model_path)
    ppnet.cuda()
    last_layer_transpose = torch.transpose(ppnet.last_layer.weight, 0, 1).detach().cpu().numpy()
    np.save(f'{save_dir}/last_layer_weight_T.npy', last_layer_transpose)
    prototype_vectors = ppnet.prototype_vectors.squeeze()
    prototype_vectors = prototype_vectors.detach().cpu().numpy()

    proto_json = json_load(proto_path)
    proto_ids = [v for v in proto_json.values()]
    extract_proto_probs(data_dir_root+'/train/', [x[:-4] for x in proto_ids], model_path, save_dir)

    features = np.load(f'{save_dir}/extracted_features.npy')
    sample_ids = [x for x in np.load(f'{save_dir}/sample_ids.npy', allow_pickle=True)] + [x for x in proto_ids]
    pred_labels = np.concatenate((np.load(f'{save_dir}/labels.npy'), np.argmax(last_layer_transpose, axis=1)))
    features = np.concatenate((features, prototype_vectors), axis=0)
    logits = np.load(f'{save_dir}/logits_marginlesses.npy')
    proto_logits = np.load(f'{save_dir}/proto_logits_marginlesses.npy')

    probs = F.softmax(torch.from_numpy(logits), dim=1).numpy()
    proto_probs = F.softmax(torch.from_numpy(proto_logits), dim=1).numpy()

    probs = np.concatenate((probs, proto_probs), axis=0)

    embedding = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, distance='angular') 
    X_transformed = embedding.fit_transform(features, init="pca")

    proto_json = json_load(proto_path)
    proto_class = np.array_split(np.argmax(last_layer_transpose, axis=1), 6)

    ########################Creating CSV########################
    print('Creating CSV', flush=True)
    predictions = np.concatenate((np.load(f'{save_dir}/predictions.npy'), np.argmax(last_layer_transpose, axis=1)))
    np.save(f'{save_dir}/X_transformed.npy', X_transformed)

    vote_dict_raw = pkl.load(open(f'{data_dir}/votes_dict_raw.pkl', 'rb'))

    samples_col = sample_ids
    coord_x_col = X_transformed[:, 0]
    coord_y_col = X_transformed[:, 1]
    preds_col = predictions
    proto_col = ['Yes' if x[:-4] in proto_ids else 'No' for x in sample_ids]

    sample_votes_col = [vote_dict_raw[x[:-4]] for x in sample_ids if x not in proto_ids]

    proto_votes_col = []
    src_dir = "/usr/xtmp/zg78/proto_proj/data/redownloaddata/"
    for fn in tqdm(proto_ids):
        mat = loadmat(f'{src_dir}/{fn[:-4]}.mat')
        vote = np.asarray(mat['votes']).squeeze()
        proto_votes_col.append(vote)

    votes_col = sample_votes_col + proto_votes_col

    vis_df_ = pd.DataFrame({'sample_ids': samples_col, 'coord_x': coord_x_col, 'coord_y':coord_y_col, 'predictions': preds_col, 'expert votes': votes_col, 'prototype_or_not':proto_col})

    for i in range(probs.shape[1]):
        vis_df_[f'class_{i}_prob'] = probs[:, i]

    vis_df_.to_csv(f'{save_dir}/{csv_name}_before_merge.csv', index=False)

    top_class_info = pd.read_csv(f'{save_dir}/top_class_info.csv')
    top_class_info = top_class_info.rename(columns={"sample_id": "sample_ids"})

    vis_df_['sample_ids'] = [x[:-4] for x in vis_df_['sample_ids']]
    vis_df = pd.merge(vis_df_, top_class_info, on='sample_ids', how='left')

    vis_df.to_csv(f'{save_dir}/{csv_name}.csv', index=False)
    print('Finished CSV', flush=True)

    # colors =  ['red','mediumblue','orange','navajowhite','khaki','green']

    # fig = plt.figure(figsize=(20, 20))
    # plt.scatter(X_transformed[:, 0][pred_labels == 0], X_transformed[:, 1][pred_labels == 0], s=3, alpha=1, c=colors[0], label='Other')
    # plt.scatter(X_transformed[:, 0][pred_labels == 1], X_transformed[:, 1][pred_labels == 1], s=3, alpha=1, c=colors[1], label='Seizure')
    # plt.scatter(X_transformed[:, 0][pred_labels == 2], X_transformed[:, 1][pred_labels == 2], s=3, alpha=1, c=colors[2], label='LPD')
    # plt.scatter(X_transformed[:, 0][pred_labels == 3], X_transformed[:, 1][pred_labels == 3], s=3, alpha=1, c=colors[3], label='GPD')
    # plt.scatter(X_transformed[:, 0][pred_labels == 4], X_transformed[:, 1][pred_labels == 4], s=3, alpha=1, c=colors[4], label='LRDA')
    # plt.scatter(X_transformed[:, 0][pred_labels == 5], X_transformed[:, 1][pred_labels == 5], s=3, alpha=1, c=colors[5], label='GRDA')

    # lgnd = plt.legend(fontsize=20)
    # lgnd.legendHandles[0]._sizes = [30]
    # lgnd.legendHandles[1]._sizes = [30]
    # lgnd.legendHandles[2]._sizes = [30]
    # lgnd.legendHandles[3]._sizes = [30]
    # lgnd.legendHandles[4]._sizes = [30]
    # lgnd.legendHandles[5]._sizes = [30]

    # plt.savefig(f'{save_dir}/{csv_name}.jpg')
    # plt.close('all')
