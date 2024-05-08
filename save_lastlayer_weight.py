import gc
import numpy as np
import random
import os
import faiss
import datetime
from itertools import product
from annoy import AnnoyIndex
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

if __name__ == "__main__":
    print('argparse', flush=True)
    args, save_dir, proto_path, data_dir, data_dir_root, model_path, csv_name = prase_args()

    print('Generating prototype vectors', flush=True)
    ppnet = torch.load(model_path)
    ppnet.cuda()
    last_layer_transpose = torch.transpose(ppnet.last_layer.weight, 0, 1).detach().cpu().numpy()
    
    np.save(f'{save_dir}/last_layer_weight.npy', last_layer_transpose)
