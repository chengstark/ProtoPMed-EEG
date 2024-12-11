import torch
import numpy as np
import os
import time
import json
from helpers import makedir


def push_prototypes(dataloader,
                    prototype_network_parallel,
                    preprocess_input_function=None,
                    root_dir_for_saving_prototypes=None,
                    epoch_number=None,
                    log=print):
    """
    Push each prototype to the nearest patch in the training set.

    Args:
        dataloader: PyTorch DataLoader (must be unnormalized in [0,1]).
        prototype_network_parallel: PyTorch network with prototype vectors.
        preprocess_input_function: Function to preprocess input, if required.
        root_dir_for_saving_prototypes: Directory to save prototypes, if provided.
        epoch_number: Epoch number for saving prototypes. If not provided, prototypes overwrite previous ones.
        log: Logging function.
    """
    prototype_network_parallel.eval()
    log('\tpush')

    start = time.time()
    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_network_parallel.module.num_prototypes
    num_classes = prototype_network_parallel.module.num_classes

    if root_dir_for_saving_prototypes is not None:
        if epoch_number is not None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes, f'epoch-{epoch_number}')
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    global_max_proto_act = np.full(n_prototypes, -np.inf)  # Max activation seen so far
    global_max_fmap_patches = np.zeros([
        n_prototypes,
        prototype_shape[1],
        prototype_shape[2],
        prototype_shape[3]
    ])
    global_max_proto_filenames_dict = {k: '' for k in range(n_prototypes)}

    log('\tExecuting push ...')
    for _, (search_batch_input, search_y, patient_id) in enumerate(dataloader):
        """
        Process each batch to update prototypes.
        """
        update_prototypes_on_batch(
            search_batch_input,
            prototype_network_parallel,
            global_max_proto_act,
            global_max_proto_filenames_dict,
            global_max_fmap_patches,
            patient_id=patient_id,
            search_y=search_y,
            num_classes=num_classes,
            preprocess_input_function=preprocess_input_function
        )

    prototype_update = np.reshape(global_max_fmap_patches, tuple(prototype_shape))
    prototype_network_parallel.module.prototype_vectors.data.copy_(
        torch.tensor(prototype_update, dtype=torch.float32).cuda()
    )

    end = time.time()
    log(f'\tpush time:\t{end - start}')

    if proto_epoch_dir is not None:
        json.dump(global_max_proto_filenames_dict, open(f'{proto_epoch_dir}/global_max_proto_filenames_dict.json', 'w'))


def update_prototypes_on_batch(search_batch_input,
                               prototype_network_parallel,
                               global_max_proto_act,
                               global_max_proto_filenames_dict,
                               global_max_fmap_patches,
                               patient_id=None,
                               search_y=None,
                               num_classes=None,
                               preprocess_input_function=None):
    """
    Update each prototype for the current search batch.

    Args:
        search_batch_input: Input batch for searching.
        prototype_network_parallel: PyTorch network with prototype vectors.
        global_max_proto_act: Max activation per prototype (updated in place).
        global_max_proto_filenames_dict: Dict of filenames for max activations (updated in place).
        global_max_fmap_patches: Feature map patches for max activations (updated in place).
        patient_id: IDs of patients in the batch.
        search_y: Ground truth labels for the batch.
        num_classes: Number of classes.
        preprocess_input_function: Function to preprocess input, if required.
    """
    prototype_network_parallel.eval()

    if preprocess_input_function is not None:
        search_batch = preprocess_input_function(search_batch_input)
    else:
        search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()
        protoL_input_torch, proto_act_torch = prototype_network_parallel.module.push_forward(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_act_ = np.copy(proto_act_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_act_torch

    class_to_img_index_dict = {key: [] for key in range(num_classes)}
    for img_index, img_y in enumerate(search_y):
        img_label = img_y.item()
        class_to_img_index_dict[img_label].append(img_index)

    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_shape[0]

    for j in range(n_prototypes):
        proto_act_j = proto_act_[:, j, :, :]
        batch_max_proto_act_j = np.amax(proto_act_j)
        if batch_max_proto_act_j > global_max_proto_act[j]:
            batch_argmax_proto_act_j = list(np.unravel_index(np.argmax(proto_act_j, axis=None), proto_act_j.shape))

            img_index_in_batch = batch_argmax_proto_act_j[0]

            fmap_height_start_index = batch_argmax_proto_act_j[1]
            fmap_height_end_index = fmap_height_start_index + prototype_shape[2]
            fmap_width_start_index = batch_argmax_proto_act_j[2]
            fmap_width_end_index = fmap_width_start_index + prototype_shape[3]

            batch_max_fmap_patch_j = protoL_input_[img_index_in_batch,
                                                   :,
                                                   fmap_height_start_index:fmap_height_end_index,
                                                   fmap_width_start_index:fmap_width_end_index]

            global_max_fmap_patches[j] = batch_max_fmap_patch_j
            global_max_proto_act[j] = batch_max_proto_act_j
            global_max_proto_filenames_dict[j] = patient_id[img_index_in_batch]

    del class_to_img_index_dict
