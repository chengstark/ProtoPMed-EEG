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
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[...,::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    #plt.imshow(img_rgb_float)
    #plt.axis('off')
    plt.imsave(fname, img_rgb_float)

class ImagePatch:

    def __init__(self, patch, label, activation,
                 sample_id=None, conv_output=None):
        self.patch = patch
        self.label = label
        self.conv_output = conv_output
        self.activation = activation

        self.sample_id = sample_id

    def __lt__(self, other):
        return self.activation < other.activation


class ImagePatchInfo:

    def __init__(self, label, activation):
        self.label = label
        self.activation = activation

    def __lt__(self, other):
        return self.activation < other.activation


# find the nearest patches in the dataset to each prototype
def find_k_nearest_patches_to_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                                         dataloader_dir, # dir for where the signals are
                                         prototype_network_parallel, # pytorch network with prototype_vectors
                                         k=5,
                                         preprocess_input_function=None, # normalize if needed
                                         full_save=False, # save all the images
                                         root_dir_for_saving_signals='./nearest',
                                         log=print):
    prototype_network_parallel.eval()
    '''
    full_save=False will only return the class identity of the closest
    patches, but it will not save anything.
    '''
    log('find nearest patches')
    start = time.time()
    n_prototypes = prototype_network_parallel.module.num_prototypes
    
    heaps = []
    # allocate an array of n_prototypes number of heaps
    for _ in range(n_prototypes):
        # a heap in python is just a maintained list
        heaps.append([])

    for idx, (search_batch_input, search_y, sample_id) in enumerate(tqdm(dataloader)):
        print('batch {}'.format(idx))
        if preprocess_input_function is not None:
            # print('preprocessing input for pushing ...')
            # search_batch = copy.deepcopy(search_batch_input)
            search_batch = preprocess_input_function(search_batch_input[:, :3, : , :])

        else:
            search_batch = search_batch_input

        with torch.no_grad():
            search_batch = search_batch.cuda()
            protoL_input_torch, proto_act_torch = \
                prototype_network_parallel.module.push_forward(search_batch)

        #protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
        proto_act_ = np.copy(proto_act_torch.detach().cpu().numpy())

        for img_idx, act_map in enumerate(proto_act_):
            for j in range(n_prototypes):
                # find the closest patches in this batch to prototype j
                closest_patch_activation_to_prototype_j = np.amax(act_map[j])

                if full_save:
                    most_activated_patch_in_act_map_j = \
                        list(np.unravel_index(np.argmax(act_map[j],axis=None),
                                              act_map[j].shape))
                    most_activated_patch_in_act_map_j = [0] + most_activated_patch_in_act_map_j
                    
                    closest_patch = search_batch_input[img_idx, :, :]
                    closest_patch = closest_patch.numpy()

                    # construct the closest patch object
                    closest_patch = ImagePatch(patch=closest_patch,
                                               label=search_y[img_idx],
                                               conv_output=protoL_input_torch[img_idx],
                                               activation=closest_patch_activation_to_prototype_j,
                                               sample_id=sample_id[img_idx])
                else:
                    closest_patch = ImagePatchInfo(label=search_y[img_idx],
                                                   activation=closest_patch_activation_to_prototype_j)


                # add to the j-th heap
                if len(heaps[j]) < k:
                    heapq.heappush(heaps[j], closest_patch)
                else:
                    # heappushpop runs more efficiently than heappush
                    # followed by heappop
                    heapq.heappushpop(heaps[j], closest_patch)

    proto_sample_id_dict = dict()

    # after looping through the dataset every heap will
    # have the k closest prototypes
    for j in tqdm(range(n_prototypes)):
        # finally sort the heap; the heap only contains the k closest
        # but they are not ranked yet
        heaps[j].sort()
        heaps[j] = heaps[j][::-1]
        proto_sample_id_dict[j] = []
        if full_save:

            dir_for_saving_signals = os.path.join(root_dir_for_saving_signals, 'proto-'+str(j))
            makedir(dir_for_saving_signals)

            labels = []

            for i, patch in enumerate(heaps[j]):
                # FIXME: for each prototype use save_signal_visualization from dataHelper.py
                # save_signal_visualization(dataloader_dir, patch.sample_id, os.path.join(dir_for_saving_signals, 'nearest-'+str(i)+'-class-'+str(patch.label)), figsize=(10, 5))
                proto_sample_id_dict[j].append(patch.sample_id)
            
            labels = np.array([patch.label for patch in heaps[j]])
            np.save(os.path.join(dir_for_saving_signals, 'class_id.npy'), labels)

    labels_all_prototype = np.array([[patch.label for patch in heaps[j]] for j in range(n_prototypes)])
    acts_all_prototype = np.array([[patch.activation for patch in heaps[j]] for j in range(n_prototypes)])

    if full_save:
        np.save(os.path.join(root_dir_for_saving_signals, 'full_class_id.npy'), labels_all_prototype)
        np.save(os.path.join(root_dir_for_saving_signals, 'full_class_act.npy'), acts_all_prototype)

    end = time.time()
    log('\tfind nearest patches time: \t{0}'.format(end - start))

    return labels_all_prototype, proto_sample_id_dict
