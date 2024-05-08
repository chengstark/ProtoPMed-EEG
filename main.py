import os
import shutil
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use("Agg")
import torch
import torch.utils.data

import argparse
from dataHelper import EEGDataset, EEGDataset_CV
from helpers import makedir
import model
import push
import train_and_test as tnt
import save
from log import create_logger
import random
from datetime import datetime

from eeg_model_features import _DenseBlock, _DenseLayer, DenseNetClassifier, _Transition, DenseNetEnconder


parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
parser.add_argument('-experiment_run', nargs=1, type=str, default='0')
parser.add_argument("-latent", nargs=1, type=int, default=32)
parser.add_argument("-last_layer_weight", nargs=1, type=int, default=None)
parser.add_argument("-model", type=str)
parser.add_argument("-base", type=str, default='eeg_model')
parser.add_argument("-random_seed", nargs=1, type=int)
parser.add_argument("-num_workers", nargs=1, type=int, default=2)
parser.add_argument("-latent_space_type", nargs=1, type=str, default='l2')
parser.add_argument("-add_on_layers_type", nargs=1, type=str, default='regular')
parser.add_argument("-model_dir_root", nargs=1, type=str)
parser.add_argument("-m", type=float)
parser.add_argument("-CV_fold", type=str)
parser.add_argument("-singe_class_prototype_per_class", type=int)
parser.add_argument("-joint_prototypes_per_border", type=int)


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
experiment_run = args.experiment_run[0]
load_model_dir = args.model
base_architecture = args.base
last_layer_weight = args.last_layer_weight[0]
num_workers = args.num_workers
model_dir_root = args.model_dir_root[0]
random_seed_number = args.random_seed[0]
add_on_layers_type = args.add_on_layers_type[0]
latent_space_type = args.latent_space_type[0]
m = args.m
CV_fold = 'all' if args.CV_fold == 'all' else int(args.CV_fold)
singe_class_prototype_per_class = args.singe_class_prototype_per_class
joint_prototypes_per_border = args.joint_prototypes_per_border

torch.manual_seed(random_seed_number)
torch.cuda.manual_seed(random_seed_number)
np.random.seed(random_seed_number)
random.seed(random_seed_number)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

# book keeping namings and code
if base_architecture is None:
    from settings import base_architecture

model_dir =  model_dir_root + '/' + base_architecture + '/' + datetime.now().strftime("%m-%d-%Y_%H-%M") + '_' + experiment_run + '_' + datetime.now().strftime("%s") + '/'
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'push.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'dataHelper.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'eeg_model_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))

proto_dir = os.path.join(model_dir, 'protos')
makedir(proto_dir)
weight_matrix_filename = 'outputL_weights'
prototype_filename_prefix = 'prototype'

# load the settings
from settings import train_dir, test_dir, train_push_dir, \
    train_batch_size, test_batch_size, train_push_batch_size, \
    proto_dim, proto_depth, \
    num_classes, prototype_activation_function, width

# all datasets
# train set
train_dataset = EEGDataset_CV(train_dir, aug=True)
if CV_fold == 'all':
    train_dataset.set_all()
else:
    train_dataset.set_train(CV_fold)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=False)

# push set
train_push_dataset = EEGDataset_CV(train_push_dir)
train_push_dataset.set_all()
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=False)

# test set
if CV_fold == 'all':
    test_dataset = EEGDataset_CV(test_dir)
    test_dataset.set_all()
else:
    test_dataset = train_dataset
    test_dataset.set_val(CV_fold)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=False)

# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log("proto_dim: {0}".format(proto_dim))
log("singe_class_prototype_per_class: {0}".format(singe_class_prototype_per_class))
log("joint_prototypes_per_border: {0}".format(joint_prototypes_per_border))
log("proto_depth: {0}".format(proto_depth))
log('margin size: {0}'.format(m))
log("saving models to: {0}".format(model_dir))
log('training set location: {0}'.format(train_dir))
log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set location: {0}'.format(train_push_dir))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set location: {0}'.format(test_dir))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))
log('eeg model width: {0}'.format(width))
log('CV fold: {0}'.format(CV_fold))
log('singe_class_prototype_per_class: {0}'.format(singe_class_prototype_per_class))
log('joint_prototypes_per_border: {0}'.format(joint_prototypes_per_border))



# construct the model
if load_model_dir:
    ppnet = torch.load(load_model_dir)
    log('starting from model: {0}'.format(load_model_dir))
else:    
    ppnet = model.construct_PPNet(pretrained=True,
                    proto_dim=proto_dim, singe_class_prototype_per_class=singe_class_prototype_per_class, 
                    joint_prototypes_per_border=joint_prototypes_per_border, proto_depth=proto_depth,
                    num_classes=num_classes, prototype_activation_function=prototype_activation_function, 
                    last_layer_weight=last_layer_weight, add_on_layers_type=add_on_layers_type,
                    class_specific=True, m=m, latent_space_type=latent_space_type, width=width)

ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)

# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size
joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

from settings import warm_optimizer_lrs
warm_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

from settings import last_layer_optimizer_lr
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

# weighting of different training losses
from settings import coefs
log(f'Coefs: {coefs}')

# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs

# train the model
log('start training')

train_acc = []
test_acc = []
currbest, best_epoch = 0, -1
currbest_seizure_roc_auc, best_seizure_epoch = 0, -1

for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        train_acc_, _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer, coefs=coefs, log=log, epoch=epoch)
    else:
        tnt.joint(model=ppnet_multi, log=log)
        train_acc_, _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer, coefs=coefs, log=log, epoch=epoch)
        joint_lr_scheduler.step()

    acc, test_seizure_roc_auc = tnt.test(model=ppnet_multi, dataloader=test_loader, log=log, epoch=epoch)
    if epoch % 10 == 0:
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=acc,
                                    target_accu=0.00, log=log)

    train_acc.append(train_acc_)
    if currbest < acc:
        currbest = acc
        best_epoch = epoch
    log("\tcurrent best acc is: \t\t{} at epoch {}".format(currbest, best_epoch))
    if currbest_seizure_roc_auc < test_seizure_roc_auc:
        currbest_seizure_roc_auc = test_seizure_roc_auc
        best_seizure_epoch = epoch
    log("\tcurrent best seizure roc auc is: \t\t{} at epoch {}".format(currbest_seizure_roc_auc, best_seizure_epoch))
    test_acc.append(acc)
    plt.plot(train_acc, "b", label="train")
    plt.plot(test_acc, "r", label="test")
    plt.ylim(0.2, 0.8)
    plt.legend()
    plt.savefig(model_dir + 'train_test_acc.png')
    plt.close()


    if epoch >= push_start and epoch in push_epochs:
        push.push_prototypes(
            train_push_loader,
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            preprocess_input_function=None, # normalize if needed
            root_dir_for_saving_prototypes=proto_dir, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            log=log)
        accu, test_seizure_roc_auc = tnt.test(model=ppnet_multi, dataloader=test_loader, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                    target_accu=0.00, log=log)

        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(10):
                log('iteration: \t{0}'.format(i))
                train_acc_, _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer, coefs=coefs, log=log)
                test_acc_, test_seizure_roc_auc = tnt.test(model=ppnet_multi, dataloader=test_loader, log=log)
                save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=test_acc_,
                                            target_accu=0.00, log=log)
                train_acc.append(train_acc_)
                test_acc.append(test_acc_)

                if currbest < test_acc_:
                    currbest = test_acc_
                    best_epoch = epoch

                if currbest_seizure_roc_auc < test_seizure_roc_auc:
                    currbest_seizure_roc_auc = test_seizure_roc_auc
                    best_seizure_epoch = epoch

                plt.plot(train_acc, "b", label="train")
                plt.plot(test_acc, "r", label="test")
                plt.ylim(0.2, 0.8)
                plt.legend()
                plt.savefig(model_dir + 'train_test_acc' + ".png")
                plt.close()
   
logclose()

