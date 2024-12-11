import os
import json
import numpy as np
import torch

def silent_print(string):
    """
    Suppress printing of a string.

    Args:
        string (str): The string to suppress.
    """
    pass

def makedir(path):
    """
    Create a directory if it does not already exist.

    Args:
        path (str): Path to the directory to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def print_and_write(string, file):
    """
    Print a string and write it to a file.

    Args:
        string (str): The string to print and write.
        file (file object): The file to write the string to.
    """
    print(string)
    file.write(string + '\n')

def find_high_activation_crop(activation_map, percentile=95):
    """
    Find the bounding box of the high-activation area in a 2D activation map.

    Args:
        activation_map (np.ndarray): 2D array of activation values.
        percentile (float, optional): Percentile threshold for activation. Defaults to 95.

    Returns:
        tuple: Coordinates of the bounding box (lower_y, upper_y, lower_x, upper_x).
    """
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0

    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break

    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break

    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > 0.5:
            lower_x = j
            break

    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > 0.5:
            upper_x = j
            break

    return lower_y, upper_y + 1, lower_x, upper_x + 1

def json_load(filepath):
    """
    Load a JSON file and convert keys to integers.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict: Dictionary with integer keys.
    """
    with open(filepath, 'r') as f:
        dict_ = json.load(f)
    return {int(k): v for k, v in dict_.items()}


def save_model_w_condition(model, model_dir, model_name, accu, target_accu, log=print):
    '''
    model: this is not the multigpu model
    '''
    if accu > target_accu:
        log('\tabove {0:.2f}%'.format(target_accu * 100))
        # torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
