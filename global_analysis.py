import torch
import torch.utils.data
from dataHelper import EEGDataset, save_signal_visualization
import json
import re
import os

from helpers import makedir, json_load
import find_nearest

import argparse

# Usage: python3 global_analysis.py -modeldir='./saved_models/' -model=''
parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-model_dir', nargs=1, type=str)
parser.add_argument('-model', nargs=1, type=str)
parser.add_argument('-test_dir', nargs=1, type=str)
parser.add_argument('-push_dir', nargs=1, type=str)
parser.add_argument('-train_dir', nargs=1, type=str)

# parser.add_argument('-dataset', nargs=1, type=str, default='cub200')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
load_model_dir = args.model_dir[0]
load_model_name = args.model[0]

load_model_path = os.path.join(load_model_dir, load_model_name)
epoch_number_str = re.search(r'\d+', load_model_name).group(0)
start_epoch_number = int(epoch_number_str)

# load the model
print('load model from ' + load_model_path)
ppnet = torch.load(load_model_path)
# print('convert to maxpool logic')
# ppnet.set_topk_k(1)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)

# load the data
# must use unaugmented (original) dataset
train_dir = args.train_dir[0]
push_dir = args.push_dir[0]
test_dir = args.test_dir[0]

batch_size = 100

# train set: do not normalize
# train set
train_dataset = EEGDataset(train_dir)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=2, pin_memory=False)

# push set
train_push_dataset = EEGDataset(push_dir)
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=batch_size, shuffle=False,
    num_workers=2, pin_memory=False)

# test set
test_dataset = EEGDataset(test_dir)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False,
    num_workers=2, pin_memory=False)

root_dir_for_saving_train_signals = os.path.join(load_model_dir,
                                                load_model_name.split('.pth')[0] + '_nearest_train')
root_dir_for_saving_test_signals = os.path.join(load_model_dir,
                                                load_model_name.split('.pth')[0] + '_nearest_test')
makedir(root_dir_for_saving_train_signals)
makedir(root_dir_for_saving_test_signals)

# save prototypes in original images

proto_epoch_dir = os.path.join(os.path.join(load_model_dir, 'protos'), 'epoch-'+str(start_epoch_number))
global_max_proto_filenames_dict = json_load(f'{proto_epoch_dir}/global_max_proto_filenames_dict.json')

for j in range(ppnet.num_prototypes):
    makedir(os.path.join(root_dir_for_saving_train_signals, str(j)))
    makedir(os.path.join(root_dir_for_saving_test_signals, str(j)))
    # FIXME: for each prototype use save_signal_visualization from dataHelper.py
    # save_signal_visualization(push_dir, global_max_proto_filenames_dict[j], os.path.join(root_dir_for_saving_train_signals, 'proto-'+str(j)), figsize=(10, 5))
    # save_signal_visualization(push_dir, global_max_proto_filenames_dict[j], os.path.join(root_dir_for_saving_test_signals, 'proto-'+str(j)), figsize=(10, 5)) # might be unnecessary

k = 10

train_labels_all_prototype, train_proto_sample_id_dict = find_nearest.find_k_nearest_patches_to_prototypes(
    dataloader=train_loader, # pytorch dataloader (must be unnormalized in [0,1])
    dataloader_dir=train_dir,
    prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
    k=k+1,
    preprocess_input_function=None, # normalize if needed
    full_save=True,
    root_dir_for_saving_signals=root_dir_for_saving_train_signals,
    log=print)

print(train_labels_all_prototype)

test_labels_all_prototype, test_proto_sample_id_dict = find_nearest.find_k_nearest_patches_to_prototypes(
    dataloader=test_loader, # pytorch dataloader (must be unnormalized in [0,1])
    dataloader_dir=test_dir,
    prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
    k=k,
    preprocess_input_function=None, # normalize if needed
    full_save=True,
    root_dir_for_saving_signals=root_dir_for_saving_test_signals,
    log=print)
print(test_labels_all_prototype)

print("last layer transpose: \n", torch.transpose(ppnet.last_layer.weight, 0, 1))

json.dump(train_proto_sample_id_dict, open(f'{root_dir_for_saving_train_signals}/proto_sample_id_dict.json', 'w'))
json.dump(test_proto_sample_id_dict, open(f'{root_dir_for_saving_test_signals}/proto_sample_id_dict.json', 'w'))

print("see analysis in ", root_dir_for_saving_train_signals)
print("see analysis in ", root_dir_for_saving_test_signals)