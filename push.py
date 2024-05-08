import torch
import numpy as np
import os
import time
import json
from helpers import makedir


# push each prototype to the nearest patch in the training set
def push_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel, # pytorch network with prototype_vectors
                    preprocess_input_function=None, # normalize if needed
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    log=print):

    prototype_network_parallel.eval()
    log('\tpush')

    start = time.time()
    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_network_parallel.module.num_prototypes
    num_classes = prototype_network_parallel.module.num_classes

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes, 'epoch-'+str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    global_max_proto_act = np.full(n_prototypes, -np.inf) # max activation seen so far
    # saves the patch representation that gives the current max activation
    global_max_fmap_patches = np.zeros(
        [n_prototypes,
         prototype_shape[1],
         prototype_shape[2],
         prototype_shape[3]])
    global_max_proto_filenames_dict = dict()
    for k in range(n_prototypes):
        global_max_proto_filenames_dict[k] = ''
    
    log('\tExecuting push ...')
    for _, (search_batch_input, search_y, patient_id) in enumerate(dataloader):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        update_prototypes_on_batch(search_batch_input,
                                   prototype_network_parallel,
                                   global_max_proto_act,
                                   global_max_proto_filenames_dict,
                                   global_max_fmap_patches,
                                   patient_id=patient_id,
                                   search_y=search_y,
                                   num_classes=num_classes,
                                   preprocess_input_function=preprocess_input_function)

    prototype_update = np.reshape(global_max_fmap_patches, tuple(prototype_shape))
    prototype_network_parallel.module.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())

    end = time.time()
    log('\tpush time: \t{0}'.format(end -  start))

    # This is where prototype image is saved in the fixed version
    json.dump(global_max_proto_filenames_dict, open(f'{proto_epoch_dir}/global_max_proto_filenames_dict.json', 'w'))

# update each prototype for current search batch
def update_prototypes_on_batch(search_batch_input,
                               prototype_network_parallel,
                               global_max_proto_act, # this will be updated
                               global_max_proto_filenames_dict, # this will be updated
                               global_max_fmap_patches, # this will be updated
                               patient_id=None,
                               search_y=None, # required if class_specific == True
                               num_classes=None, # required if class_specific == True
                               preprocess_input_function=None):

    prototype_network_parallel.eval()

    if preprocess_input_function is not None:
        search_batch = preprocess_input_function(search_batch_input)
    else:
        search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()
        # this computation currently is not parallelized
        protoL_input_torch, proto_act_torch = prototype_network_parallel.module.push_forward(search_batch)
    
    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_act_ = np.copy(proto_act_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_act_torch

    class_to_img_index_dict = {key: [] for key in range(num_classes)}
    # img_y is the image's integer label
    for img_index, img_y in enumerate(search_y):
        img_label = img_y.item()
        class_to_img_index_dict[img_label].append(img_index)

    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_shape[0]

    for j in range(n_prototypes):        
        # target_class is the class of the class_specific prototype
        # target_class = torch.argmax(prototype_network_parallel.module.prototype_class_identity[j]).item()

        # if there is not images of the target_class from this batch
        # we go on to the next prototype
        # if len(class_to_img_index_dict[target_class]) == 0:
        #     continue
        # proto_act_j = proto_act_[class_to_img_index_dict[target_class]][:,j,:,:] # class specific push
        proto_act_j = proto_act_[:,j, :, :] # class unspecific push
        batch_max_proto_act_j = np.amax(proto_act_j)
        if batch_max_proto_act_j > global_max_proto_act[j]:
            batch_argmax_proto_act_j = \
                list(np.unravel_index(np.argmax(proto_act_j, axis=None), proto_act_j.shape))
            # img_index_in_batch = class_to_img_index_dict[target_class][batch_argmax_proto_act_j[0]] # class specific push

            img_index_in_batch = batch_argmax_proto_act_j[0] # class unspecific push

            # Set batch_max_fmap_patch_j to prototype vector
            fmap_height_start_index = batch_argmax_proto_act_j[1]
            fmap_height_end_index = fmap_height_start_index + prototype_shape[2]
            fmap_width_start_index = batch_argmax_proto_act_j[2]
            fmap_width_end_index = fmap_width_start_index + prototype_shape[3]

            batch_max_fmap_patch_j = protoL_input_[img_index_in_batch,
                                                   :,
                                                   fmap_height_start_index:fmap_height_end_index, # for whole signal, could change to ':'
                                                   fmap_width_start_index:fmap_width_end_index] # for whole signal, could change to ':'

            # updates current best prototypes in this epoch so far
            global_max_fmap_patches[j] = batch_max_fmap_patch_j
            global_max_proto_act[j] = batch_max_proto_act_j            
            global_max_proto_filenames_dict[j] = patient_id[img_index_in_batch]
            

    del class_to_img_index_dict
