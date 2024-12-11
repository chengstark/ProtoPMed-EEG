import torch
import numpy as np
import heapq
import matplotlib.pyplot as plt
import os
import time
from dataHelper import save_signal_visualization
import cv2
from helpers import makedir
from tqdm import tqdm

def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                     bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    """
    Save an image with a bounding box.

    Args:
        fname (str): File name to save the image.
        img_rgb (np.ndarray): RGB image.
        bbox_height_start (int): Starting height of the bounding box.
        bbox_height_end (int): Ending height of the bounding box.
        bbox_width_start (int): Starting width of the bounding box.
        bbox_width_end (int): Ending width of the bounding box.
        color (tuple): Bounding box color.
    """
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255 * img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end - 1, bbox_height_end - 1),
                  color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[..., ::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    plt.imsave(fname, img_rgb_float)

class ImagePatch:
    """
    Class to represent an image patch with associated data.

    Args:
        patch (np.ndarray): The image patch.
        label (int): Label of the patch.
        activation (float): Activation value for the patch.
        sample_id (str): Sample ID.
        conv_output (torch.Tensor): Convolution output.
    """
    def __init__(self, patch, label, activation, sample_id=None, conv_output=None):
        self.patch = patch
        self.label = label
        self.conv_output = conv_output
        self.activation = activation
        self.sample_id = sample_id

    def __lt__(self, other):
        return self.activation < other.activation

class ImagePatchInfo:
    """
    Class to represent basic image patch information.

    Args:
        label (int): Label of the patch.
        activation (float): Activation value.
    """
    def __init__(self, label, activation):
        self.label = label
        self.activation = activation

    def __lt__(self, other):
        return self.activation < other.activation

def find_k_nearest_patches_to_prototypes(dataloader, dataloader_dir, prototype_network_parallel, k=5,
                                         preprocess_input_function=None, full_save=False,
                                         root_dir_for_saving_signals='./nearest', log=print):
    """
    Find the k-nearest patches in the dataset to each prototype.

    Args:
        dataloader (DataLoader): PyTorch DataLoader (unnormalized in [0, 1]).
        dataloader_dir (str): Directory where signals are located.
        prototype_network_parallel (nn.Module): PyTorch network with prototype vectors.
        k (int): Number of nearest patches to find.
        preprocess_input_function (function): Function to preprocess input if needed.
        full_save (bool): Whether to save all images.
        root_dir_for_saving_signals (str): Directory to save nearest patches.
        log (function): Logging function.

    Returns:
        labels_all_prototype (np.ndarray): Labels of all nearest patches for each prototype.
        proto_sample_id_dict (dict): Dictionary of sample IDs for each prototype.
    """
    prototype_network_parallel.eval()
    log('Finding nearest patches')
    start = time.time()
    n_prototypes = prototype_network_parallel.module.num_prototypes

    heaps = [[] for _ in range(n_prototypes)]

    for idx, (search_batch_input, search_y, sample_id) in enumerate(tqdm(dataloader)):
        if preprocess_input_function is not None:
            search_batch = preprocess_input_function(search_batch_input[:, :3, :, :])
        else:
            search_batch = search_batch_input

        with torch.no_grad():
            search_batch = search_batch.cuda()
            protoL_input_torch, proto_act_torch = prototype_network_parallel.module.push_forward(search_batch)

        proto_act_ = np.copy(proto_act_torch.detach().cpu().numpy())

        for img_idx, act_map in enumerate(proto_act_):
            for j in range(n_prototypes):
                closest_patch_activation_to_prototype_j = np.amax(act_map[j])

                if full_save:
                    most_activated_patch_in_act_map_j = list(np.unravel_index(np.argmax(act_map[j], axis=None),
                                                                               act_map[j].shape))
                    most_activated_patch_in_act_map_j = [0] + most_activated_patch_in_act_map_j

                    closest_patch = ImagePatch(patch=search_batch_input[img_idx].numpy(),
                                               label=search_y[img_idx],
                                               conv_output=protoL_input_torch[img_idx],
                                               activation=closest_patch_activation_to_prototype_j,
                                               sample_id=sample_id[img_idx])
                else:
                    closest_patch = ImagePatchInfo(label=search_y[img_idx],
                                                   activation=closest_patch_activation_to_prototype_j)

                if len(heaps[j]) < k:
                    heapq.heappush(heaps[j], closest_patch)
                else:
                    heapq.heappushpop(heaps[j], closest_patch)

    proto_sample_id_dict = {}

    for j in tqdm(range(n_prototypes)):
        heaps[j].sort(reverse=True)
        proto_sample_id_dict[j] = []

        if full_save:
            dir_for_saving_signals = os.path.join(root_dir_for_saving_signals, f'proto-{j}')
            makedir(dir_for_saving_signals)

            for i, patch in enumerate(heaps[j]):
                proto_sample_id_dict[j].append(patch.sample_id)

            labels = np.array([patch.label for patch in heaps[j]])
            np.save(os.path.join(dir_for_saving_signals, 'class_id.npy'), labels)

    labels_all_prototype = np.array([[patch.label for patch in heaps[j]] for j in range(n_prototypes)])
    acts_all_prototype = np.array([[patch.activation for patch in heaps[j]] for j in range(n_prototypes)])

    if full_save:
        np.save(os.path.join(root_dir_for_saving_signals, 'full_class_id.npy'), labels_all_prototype)
        np.save(os.path.join(root_dir_for_saving_signals, 'full_class_act.npy'), acts_all_prototype)

    end = time.time()
    log(f'Find nearest patches time: {end - start}')

    return labels_all_prototype, proto_sample_id_dict
