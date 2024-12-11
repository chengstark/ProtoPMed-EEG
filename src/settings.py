# m = 0.05

width = 5

proto_dim = (1, 1)
# singe_class_prototype_per_class = 4
# joint_prototypes_per_border = 2
proto_depth = width * 255

num_classes = 6

prototype_activation_function = "log"
prototype_activation_function_in_numpy = prototype_activation_function

class_specific = True

# add_on_layers_type = 'regular'

data_path = '/usr/xtmp/zg78/proto_proj/data/10_train_test_split_50s_/'
train_dir = data_path + 'train/'
test_dir = data_path + 'test/'
train_push_dir = data_path + 'push_votes_leq_20/'

train_batch_size = 128
test_batch_size = 128
train_push_batch_size = 128

joint_optimizer_lrs = {'features': 2e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 2e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-3

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
    'ortho': 100.0
}

num_train_epochs = 50
num_warm_epochs = 10

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]
