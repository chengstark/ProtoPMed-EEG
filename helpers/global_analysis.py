import torch
import torch.utils.data
from dataHelper import EEGDataset, save_signal_visualization
import json
import os
import re
from helpers import makedir, json_load
import find_nearest
import argparse

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpuid', nargs=1, type=str, default='0')
    parser.add_argument('-model_dir', nargs=1, type=str)
    parser.add_argument('-model', nargs=1, type=str)
    parser.add_argument('-test_dir', nargs=1, type=str)
    parser.add_argument('-push_dir', nargs=1, type=str)
    parser.add_argument('-train_dir', nargs=1, type=str)
    return parser.parse_args()

def setup_environment(args):
    """
    Setup the environment based on command-line arguments.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        dict: Dictionary of initialized paths and parameters.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
    load_model_dir = args.model_dir[0]
    load_model_name = args.model[0]
    load_model_path = os.path.join(load_model_dir, load_model_name)
    epoch_number_str = re.search(r'\d+', load_model_name).group(0)
    start_epoch_number = int(epoch_number_str)

    return {
        'load_model_path': load_model_path,
        'start_epoch_number': start_epoch_number,
        'train_dir': args.train_dir[0],
        'push_dir': args.push_dir[0],
        'test_dir': args.test_dir[0],
        'load_model_dir': load_model_dir,
        'load_model_name': load_model_name,
    }

def load_model(load_model_path):
    """
    Load and prepare the model.

    Args:
        load_model_path (str): Path to the model.

    Returns:
        nn.Module: Loaded model.
    """
    print(f'Loading model from {load_model_path}')
    ppnet = torch.load(load_model_path).cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    return ppnet, ppnet_multi

def prepare_dataloaders(train_dir, push_dir, test_dir, batch_size):
    """
    Prepare data loaders for train, push, and test datasets.

    Args:
        train_dir (str): Path to train data.
        push_dir (str): Path to push data.
        test_dir (str): Path to test data.
        batch_size (int): Batch size for data loaders.

    Returns:
        dict: Dictionary of data loaders.
    """
    train_dataset = EEGDataset(train_dir)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=False)

    train_push_dataset = EEGDataset(push_dir)
    train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=False)

    test_dataset = EEGDataset(test_dir)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=False)

    return {
        'train_loader': train_loader,
        'train_push_loader': train_push_loader,
        'test_loader': test_loader,
    }

def save_prototypes(global_max_proto_filenames_dict, root_dir_for_saving_signals, push_dir, num_prototypes):
    """
    Save prototype visualizations.

    Args:
        global_max_proto_filenames_dict (dict): Mapping of prototypes to filenames.
        root_dir_for_saving_signals (str): Root directory for saving visualizations.
        push_dir (str): Path to push data.
        num_prototypes (int): Number of prototypes.
    """
    for j in range(num_prototypes):
        dir_path = os.path.join(root_dir_for_saving_signals, str(j))
        makedir(dir_path)

        # save_signal_visualization(push_dir, global_max_proto_filenames_dict[j], os.path.join(dir_path, f'proto-{j}'))
        pass  # Placeholder for actual visualization logic

def main():
    args = parse_args()
    env = setup_environment(args)

    ppnet, ppnet_multi = load_model(env['load_model_path'])

    dataloaders = prepare_dataloaders(env['train_dir'], env['push_dir'], env['test_dir'], batch_size=100)

    root_dir_for_saving_train_signals = os.path.join(
        env['load_model_dir'], f"{env['load_model_name'].split('.pth')[0]}_nearest_train")
    root_dir_for_saving_test_signals = os.path.join(
        env['load_model_dir'], f"{env['load_model_name'].split('.pth')[0]}_nearest_test")
    makedir(root_dir_for_saving_train_signals)
    makedir(root_dir_for_saving_test_signals)

    proto_epoch_dir = os.path.join(env['load_model_dir'], 'protos', f"epoch-{env['start_epoch_number']}")
    global_max_proto_filenames_dict = json_load(f'{proto_epoch_dir}/global_max_proto_filenames_dict.json')

    save_prototypes(global_max_proto_filenames_dict, root_dir_for_saving_train_signals, env['push_dir'], ppnet.num_prototypes)

    k = 10

    train_labels_all_prototype, train_proto_sample_id_dict = find_nearest.find_k_nearest_patches_to_prototypes(
        dataloader=dataloaders['train_loader'],
        dataloader_dir=env['train_dir'],
        prototype_network_parallel=ppnet_multi,
        k=k+1,
        preprocess_input_function=None,
        full_save=True,
        root_dir_for_saving_signals=root_dir_for_saving_train_signals,
        log=print)

    print(train_labels_all_prototype)

    test_labels_all_prototype, test_proto_sample_id_dict = find_nearest.find_k_nearest_patches_to_prototypes(
        dataloader=dataloaders['test_loader'],
        dataloader_dir=env['test_dir'],
        prototype_network_parallel=ppnet_multi,
        k=k,
        preprocess_input_function=None,
        full_save=True,
        root_dir_for_saving_signals=root_dir_for_saving_test_signals,
        log=print)
    print(test_labels_all_prototype)

    print("Last layer transpose:\n", torch.transpose(ppnet.last_layer.weight, 0, 1))

    json.dump(train_proto_sample_id_dict, open(f'{root_dir_for_saving_train_signals}/proto_sample_id_dict.json', 'w'))
    json.dump(test_proto_sample_id_dict, open(f'{root_dir_for_saving_test_signals}/proto_sample_id_dict.json', 'w'))

    print("See analysis in", root_dir_for_saving_train_signals)
    print("See analysis in", root_dir_for_saving_test_signals)

if __name__ == "__main__":
    main()
