import numpy as np
import json
import argparse
import pickle as pkl
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from scipy.stats import entropy
import random

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', type=str, default='0')
parser.add_argument('-json_dir', type=str)
parser.add_argument('-test_dir', type=str)
parser.add_argument('-train_dir', type=str)
parser.add_argument('-k', type=int, default=10)



def calculate_agreement(votes_dict, votes_dict_raw, neighbourhood):
    agreement_percentages = []
    agreement_ces = []
    agreement_percentages_per_class = {
        0:[],
        1:[],
        2:[],
        3:[],
        4:[],
        5:[],
    }
    agreement_percentages_per_class_ces = {
        0:[],
        1:[],
        2:[],
        3:[],
        4:[],
        5:[],
    }

    samples = list(neighbourhood.keys())

    for sample in samples:
        neighbours = neighbourhood[sample]
        agreement = []
        ces = []
        sample_vote = np.asarray(votes_dict[sample[:-4]])
        sample_vote_raw = np.asarray(votes_dict_raw[sample[:-4]])
        sample_vote_raw_probs = sample_vote_raw / np.sum(sample_vote_raw)
        sample_vote_raw_probs = torch.tensor(sample_vote_raw_probs).float().view(1, -1)
        for s in neighbours:
            vote = np.asarray(votes_dict[s[:-4]])
            vote_raw = np.asarray(votes_dict_raw[s[:-4]])
            vote_raw_probs = vote_raw / np.sum(vote_raw)
            vote_raw_probs = torch.tensor(vote_raw_probs).float().view(1, -1)
            ce = F.cross_entropy(sample_vote_raw_probs, vote_raw_probs).numpy()

            agreement.append(int(vote == sample_vote))
            ces.append(ce)

        agreement_ces.append(sum(ces) / len(ces))
        agreement_percentages_per_class_ces[int(sample_vote)].append(sum(ces) / len(ces))

        agreement_perc = sum(agreement) / len(agreement)
        agreement_percentages.append(agreement_perc)
        agreement_percentages_per_class[int(sample_vote)].append(agreement_perc)
    
    return agreement_percentages, agreement_percentages_per_class, agreement_ces, agreement_percentages_per_class_ces

def get_mean_upper_lower(lst):
    return [np.mean(lst), np.percentile(lst, 2.5), np.percentile(lst, 97.5)]

if __name__ == "__main__":
    args = parser.parse_args()
    
    votes_dict = pkl.load(open(f'{args.test_dir}/votes_dict.pkl', 'rb'))
    votes_dict_raw = pkl.load(open(f'{args.test_dir}/votes_dict_raw.pkl', 'rb'))
    neighbourhood = json.load(open(f'{args.json_dir}/test_neighbour_{args.k}_sample_id_dict.json'))
    print('# of samples:', len(votes_dict), flush=True)

    agreement_percentages, agreement_percentages_per_class, agreement_ces, agreement_percentages_per_class_ces = \
            calculate_agreement(votes_dict, votes_dict_raw, neighbourhood)

    agreement_percentages_bt = []
    agreement_ces_bt = []
    agreement_percentages_per_class_bt = {
        0:[],
        1:[],
        2:[],
        3:[],
        4:[],
        5:[],
    }
    agreement_percentages_per_class_ces_bt = {
        0:[],
        1:[],
        2:[],
        3:[],
        4:[],
        5:[],
    }

    for i in tqdm(range(1000)):

        np.random.seed(i)
        random.seed(i)

        subsample_agreement_percentages = random.choices(agreement_percentages, k=len(agreement_percentages))
        subsample_agreement_ces = random.choices(agreement_ces, k=len(agreement_ces))

        agreement_percentages_bt.append(np.mean(subsample_agreement_percentages))
        agreement_ces_bt.append(np.mean(subsample_agreement_ces))

        for k, v in agreement_percentages_per_class.items():
            agreement_percentages_per_class_bt[k].append(np.mean(random.choices(v, k=len(v))))

        for k, v in agreement_percentages_per_class_ces.items():
            agreement_percentages_per_class_ces_bt[k].append(np.mean(random.choices(v, k=len(v))))

    print('By Max:')
    print(get_mean_upper_lower(agreement_percentages_bt))
    for k, v in agreement_percentages_per_class_bt.items():
        print(f'\t Class {k} {get_mean_upper_lower(v)}')

    print('By Vote:')
    print(get_mean_upper_lower(agreement_ces_bt))
    for k, v in agreement_percentages_per_class_ces_bt.items():
        print(f'\t Class {k} {get_mean_upper_lower(v)}')



