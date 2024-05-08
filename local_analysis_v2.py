##### MODEL AND DATA LOADING
from random import sample
import torch
import torch.utils.data
import numpy as np
from dataHelper import EEGDataset
import numpy as np
import pickle as pkl
from helpers import json_load
from tqdm import tqdm
import torch.nn.functional as F

import os
import argparse
import pandas as pd

k=3
# specify the test image to be analyzed
parser = argparse.ArgumentParser()
parser.add_argument('-sample_root_dir', type=str)
parser.add_argument('-model_path', type=str)
args = parser.parse_args()

test_samples_dir = args.sample_root_dir + 'test/'
train_samples_dir = args.sample_root_dir + 'train/'
model_path = args.model_path
protos_dir = '/'.join(model_path.split('/')[:-1])+'/protos/'
save_dir = model_path[:-4]+f"_extracted_features/{test_samples_dir.replace('/', '_')}/"

os.makedirs(save_dir, exist_ok=True)

print('test_samples_dir', test_samples_dir, flush=True)
print('train_samples_dir', train_samples_dir, flush=True)

ppnet = torch.load(model_path)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
prototype_shape = ppnet.prototype_shape

##### SANITY CHECK
# confirm prototype class identity
train_vote_dict = pkl.load(open(train_samples_dir+'/votes_dict.pkl', 'rb'))
prototype_info = json_load('/'.join(model_path.split('/')[:-1])+'/protos/epoch-'+model_path.split('/')[-1].split('push')[0].split('_')[0]+'/global_max_proto_filenames_dict.json')
prototype_sample_identity = [train_vote_dict[v[:-4]] for v in prototype_info.values()]
num_classes = len(set(prototype_sample_identity))

print('Prototypes are chosen from ' + str(len(set(prototype_sample_identity))) + ' number of classes.', flush=True)
print('Their class identities are: ' + str(prototype_sample_identity), flush=True)

# confirm prototype connects most strongly to its own class
prototype_max_connection = torch.argmax(ppnet.last_layer.weight, dim=0)
prototype_max_connection = prototype_max_connection.cpu().numpy()
if np.sum(prototype_max_connection == prototype_sample_identity) == ppnet.num_prototypes:
    print('All prototypes connect most strongly to their respective classes.', flush=True)
else:
    print('WARNING: Not all prototypes connect most strongly to their respective classes.', flush=True)

# np load

test_img_dataset = EEGDataset(test_samples_dir)
test_img_loader = torch.utils.data.DataLoader(
    test_img_dataset, batch_size=1, shuffle=False,
    num_workers=2, pin_memory=False)

top_class_info_dict = dict()
for i in range(k):
    top_class_info_dict[f'top {i+1} overall_proto index'] = []
    top_class_info_dict[f'top {i+1} overall_proto class similarity score'] = []
    top_class_info_dict[f'top {i+1} overall_proto class connection strength'] = []
    top_class_info_dict[f'top {i+1} overall_proto positive class contribution for top class'] = []

top_class_info_dict['sample_id'] = []
top_class_info_dict['similarity score vector'] = []

for i in range(k):
    top_class_info_dict[f'top {i+1} class proto index'] = []
    top_class_info_dict[f'top {i+1} class similarity score'] = []
    top_class_info_dict[f'top {i+1} class connection strength'] = []

for signal, vote, sample_id in tqdm(test_img_loader):
    top_class_info_dict['sample_id'].append(sample_id[0][:-4])

    images_test = signal.cuda()
    labels_test = vote.cuda()

    _, logits, prototype_activations = ppnet_multi(images_test)

    opccpftc_all = torch.matmul(prototype_activations, F.relu(ppnet.last_layer.weight).T).squeeze()
    top_class_info_dict['similarity score vector'].append(prototype_activations[0].detach().cpu().numpy())

    topk_logits, topk_classes = torch.topk(logits[0], k=k)
    _, top_3_proto_activation_idxs = torch.topk(prototype_activations[0], k=k)
    for i, j in enumerate(top_3_proto_activation_idxs.detach().cpu().numpy()):
        last_layer_connection = torch.max(ppnet.last_layer.weight[:, j]).detach().cpu()
        similarity_score = prototype_activations[0][j].detach().cpu()
        opccpftc = similarity_score * F.relu(ppnet.last_layer.weight[topk_classes[0], j])
        top_class_info_dict[f'top {i+1} overall_proto index'].append(j)
        top_class_info_dict[f'top {i+1} overall_proto class similarity score'].append(similarity_score.item())
        top_class_info_dict[f'top {i+1} overall_proto class connection strength'].append(last_layer_connection.item())
        top_class_info_dict[f'top {i+1} overall_proto positive class contribution for top class'].append((opccpftc / (opccpftc_all[topk_classes[0]] + 1e-8)).item())
        

    for i, c in enumerate(topk_classes.detach().cpu().numpy()):
        # :30 is selecting the top 30, pure prototypes
        class_prototype_indices = np.nonzero(ppnet.prototype_class_identity.detach().cpu().numpy()[:30, c])[0]
        class_prototype_activations = prototype_activations[0][class_prototype_indices]
        _, sorted_indices_cls_act = torch.sort(class_prototype_activations)

        j = sorted_indices_cls_act.detach().cpu().numpy()[::-1][0]
        prototype_index = class_prototype_indices[j]
        prototype_class_identity = prototype_sample_identity[prototype_index]
        last_layer_connection = ppnet.last_layer.weight[c][prototype_index]
        similarity_score = prototype_activations[0][prototype_index]
        
        top_class_info_dict[f'top {i+1} class proto index'].append(prototype_index)
        top_class_info_dict[f'top {i+1} class similarity score'].append(similarity_score.detach().cpu().item())
        top_class_info_dict[f'top {i+1} class connection strength'].append((last_layer_connection).detach().cpu().item())


top_class_info_df = pd.DataFrame(top_class_info_dict)
top_class_info_df.to_csv(f'{save_dir}/top_class_info.csv', index=False)


print(top_class_info_dict.keys())


# top overall (pure or mixed)
# top 1st proto from the first class (pure)
# top 1st proto from the second class (pure)
# top 1st proto from the third class (pure)

