import numpy as np
import json
import argparse
import pickle as pkl
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from scipy.stats import entropy
import random

def parse_arguments():
    """
    Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpuid', type=str, default='0', help="GPU ID to use.")
    parser.add_argument('-json_dir', type=str, required=True, help="Directory containing JSON files.")
    parser.add_argument('-test_dir', type=str, required=True, help="Directory containing test data.")
    parser.add_argument('-train_dir', type=str, required=True, help="Directory containing training data.")
    parser.add_argument('-k', type=int, default=10, help="Number of neighbors for agreement calculation.")
    return parser.parse_args()

def calculate_agreement(votes_dict, votes_dict_raw, neighbourhood):
    """
    Calculate agreement percentages and cross-entropy scores for given votes and neighborhood.

    Args:
        votes_dict (dict): Dictionary of votes for samples.
        votes_dict_raw (dict): Raw vote distributions for samples.
        neighbourhood (dict): Mapping of samples to their neighborhoods.

    Returns:
        tuple: Agreement percentages, percentages per class, cross-entropy scores, and CE scores per class.
    """
    agreement_percentages = []
    agreement_ces = []
    agreement_percentages_per_class = {i: [] for i in range(6)}
    agreement_percentages_per_class_ces = {i: [] for i in range(6)}

    for sample, neighbours in neighbourhood.items():
        agreement = []
        ces = []
        sample_vote = np.asarray(votes_dict[sample[:-4]])
        sample_vote_raw = np.asarray(votes_dict_raw[sample[:-4]])
        sample_vote_raw_probs = torch.tensor(sample_vote_raw / np.sum(sample_vote_raw)).float().view(1, -1)

        for neighbor in neighbours:
            vote = np.asarray(votes_dict[neighbor[:-4]])
            vote_raw = np.asarray(votes_dict_raw[neighbor[:-4]])
            vote_raw_probs = torch.tensor(vote_raw / np.sum(vote_raw)).float().view(1, -1)
            ce = F.cross_entropy(sample_vote_raw_probs, vote_raw_probs).item()

            agreement.append(int(np.array_equal(vote, sample_vote)))
            ces.append(ce)

        agreement_ces.append(np.mean(ces))
        agreement_percentages_per_class_ces[int(sample_vote)].append(np.mean(ces))
        
        agreement_perc = np.mean(agreement)
        agreement_percentages.append(agreement_perc)
        agreement_percentages_per_class[int(sample_vote)].append(agreement_perc)

    return agreement_percentages, agreement_percentages_per_class, agreement_ces, agreement_percentages_per_class_ces

def get_mean_upper_lower(lst):
    """
    Calculate mean and confidence intervals (2.5% and 97.5% percentiles).

    Args:
        lst (list): List of values.

    Returns:
        list: Mean, lower bound, and upper bound.
    """
    return [np.mean(lst), np.percentile(lst, 2.5), np.percentile(lst, 97.5)]

def main():
    """
    Main function to calculate agreement statistics and bootstrap confidence intervals.
    """
    args = parse_arguments()

    votes_dict = pkl.load(open(f'{args.test_dir}/votes_dict.pkl', 'rb'))
    votes_dict_raw = pkl.load(open(f'{args.test_dir}/votes_dict_raw.pkl', 'rb'))
    neighbourhood = json.load(open(f'{args.json_dir}/test_neighbour_{args.k}_sample_id_dict.json'))
    print('# of samples:', len(votes_dict), flush=True)

    agreement_percentages, agreement_percentages_per_class, agreement_ces, agreement_percentages_per_class_ces = \
        calculate_agreement(votes_dict, votes_dict_raw, neighbourhood)

    agreement_percentages_bt = []
    agreement_ces_bt = []
    agreement_percentages_per_class_bt = {i: [] for i in range(6)}
    agreement_percentages_per_class_ces_bt = {i: [] for i in range(6)}

    for i in tqdm(range(1000), desc="Bootstrapping"):
        np.random.seed(i)
        random.seed(i)

        subsample_agreement_percentages = random.choices(agreement_percentages, k=len(agreement_percentages))
        subsample_agreement_ces = random.choices(agreement_ces, k=len(agreement_ces))

        agreement_percentages_bt.append(np.mean(subsample_agreement_percentages))
        agreement_ces_bt.append(np.mean(subsample_agreement_ces))

        for k, v in agreement_percentages_per_class.items():
            if v:
                agreement_percentages_per_class_bt[k].append(np.mean(random.choices(v, k=len(v))))

        for k, v in agreement_percentages_per_class_ces.items():
            if v:
                agreement_percentages_per_class_ces_bt[k].append(np.mean(random.choices(v, k=len(v))))

    print('By Max:')
    print(get_mean_upper_lower(agreement_percentages_bt))
    for k, v in agreement_percentages_per_class_bt.items():
        if v:
            print(f'\t Class {k}: {get_mean_upper_lower(v)}')

    print('By Vote:')
    print(get_mean_upper_lower(agreement_ces_bt))
    for k, v in agreement_percentages_per_class_ces_bt.items():
        if v:
            print(f'\t Class {k}: {get_mean_upper_lower(v)}')

if __name__ == "__main__":
    main()
