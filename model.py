import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from eeg_model_features import eeg_model_features
import torch.nn.functional as F

import operator as op
from functools import reduce
from itertools import combinations


class PPNet(nn.Module):

    def __init__(self, features, proto_dim=(1, 1), singe_class_prototype_per_class=5, 
                    joint_prototypes_per_border=1, proto_depth=None, 
                    num_classes=3, init_weights=True, last_layer_connection_weight=None,
                 prototype_activation_function='log', m=None,
                 add_on_layers_type='bottleneck', latent_space_type='l2', 
                 class_specific=False, width=1):

        super(PPNet, self).__init__()
        self.joint_prototypes_per_border = joint_prototypes_per_border
        self.singe_class_prototype_per_class = singe_class_prototype_per_class
        self.num_classes = num_classes
        self.num_prototypes = self.singe_class_prototype_per_class * self.num_classes + self.joint_prototypes_per_border * self.ncr(self.num_classes, 2)
        self.prototype_shape = (self.num_prototypes, proto_depth, proto_dim[0], proto_dim[1])
        self.topk_k = 1 # for a 14x14: topk_k=3 is 1.5%, topk_k=9 is 4.5%
        self.class_specific=class_specific
        self.epsilon = 1e-4
        self.m = m
        self.relu_on_cos = False
        self.input_vector_length = 64
        self.last_layer_connection_weight = last_layer_connection_weight
        self.latent_space_type = latent_space_type
        
        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function

        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''

        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes, self.num_classes)
        
        for j in range(self.singe_class_prototype_per_class * self.num_classes):
            self.prototype_class_identity[j, j // self.singe_class_prototype_per_class] = 1

        # for j in range(self.singe_class_prototype_per_class * self.num_classes, self.num_prototypes):
        #     self.prototype_class_identity[j, j // self.singe_class_prototype_per_class] = 1
        #     self.prototype_class_identity[j, j // self.singe_class_prototype_per_class+j // self.singe_class_prototype_per_class] = 1
        col_index = list(combinations(range(self.num_classes), 2))
        for idx, row_index in enumerate(range(self.singe_class_prototype_per_class * self.num_classes, self.num_prototypes)):
            self.prototype_class_identity[row_index, col_index[idx % len(col_index)][0]] = 1
            self.prototype_class_identity[row_index, col_index[idx % len(col_index)][1]] = 1
        
        # this has to be named features to allow the precise loading

        self.features = features

        # features_name = str(self.features).upper()
        # if features_name.startswith('VGG') or features_name.startswith('RES'):
        #     first_add_on_layer_in_channels = \
        #         [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        # elif features_name.startswith('DENSE'):
        #     first_add_on_layer_in_channels = \
        #         [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        # else:
        #     raise Exception('other base base_architecture NOT implemented')
        first_add_on_layer_in_channels = 255 * width
        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        elif add_on_layers_type == 'identity':
            self.add_on_layers = nn.Sequential(nn.Identity())
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
                )
        
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)

        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False) # do not use bias

        if init_weights:
            self._initialize_weights()
    

    def ncr(self, n, r):
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return numer // denom

    def set_topk_k(self, topk_k):
        '''set the topk_k'''
        self.topk_k = topk_k
    
    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features(x)
        x = self.add_on_layers(x)
        return x

    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        '''
        input of shape N * c * h * w
        filter of shape P * c * h1 * w1
        weight of shape P * c * h1 * w1
        '''
        input2 = input ** 2
        input_patch_weighted_norm2 = F.conv2d(input=input2, weight=weights)

        filter2 = filter ** 2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)

        weighted_filter = filter * weights
        weighted_inner_product = F.conv2d(input=input, weight=weighted_filter)

        # use broadcast
        intermediate_result = \
            - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        batch = x.shape[0]
        # x is the conv output, shape=[Batch * channel * conv output shape]
        expanded_x = nn.Unfold(kernel_size=(self.prototype_shape[2], self.prototype_shape[3]))(x)
        expanded_x = expanded_x.unsqueeze(0).permute(0,1,3,2)
        # expanded shape = [1, batch, number of such blocks, channel*proto_shape[2]*proto_shape[3]]
        expanded_x = expanded_x.contiguous().view(1, -1, self.prototype_shape[1] *self.prototype_shape[2] * self.prototype_shape[3])
        expanded_proto = nn.Unfold(kernel_size=(self.prototype_shape[2], self.prototype_shape[3]))(self.prototype_vectors).unsqueeze(0)
        # expanded proto shape = [1, proto num, channel*proto_shape[2]*proto_shape[3], 1]
        expanded_distances = torch.cdist(expanded_x, expanded_proto.contiguous().view(1, expanded_proto.shape[1], -1))
        # [1, Batch * number of blocks in x, num proto]
        expanded_distances = torch.reshape(expanded_distances, shape=(batch, -1, self.prototype_shape[0])).permute(0,2,1)
        # print(expanded_distances.shape)
        # distances = nn.Fold(output_size=(x.shape[2] - self.prototype_shape[2] + 1, x.shape[3]- self.prototype_shape[3] + 1), kernel_size=(self.prototype_shape[2], self.prototype_shape[3]))(expanded_distances)
        distances = torch.reshape(expanded_distances, shape=(batch, self.prototype_shape[0], x.shape[2] - self.prototype_shape[2] + 1, x.shape[3] - self.prototype_shape[3] + 1))
        # distance shape = [batch, proto num, conv output shape]
        return distances

    def cos_activation(self, x, prototypes_of_wrong_class=None):
        
        is_train=True
        additive_margin=False

        '''
        Takes convolutional features and gives arc distance as in
        https://arxiv.org/pdf/1801.07698.pdf
        '''
        # Sometimes, x manages to have a length of 0
        # We add some epsilon to prevent nans from this
        input_vector_length = self.input_vector_length
        '''
        This needs to be square root of window size, since conceptually using convolution
        for this is equivalent to stacking the prototype components and dotting that with
        the stacked x window. The length of such a stacked vector, if the components are normalized,
        is sqrt(n) where n is the number of vectors being stacked.
        '''
        normalizing_factor = (self.prototype_shape[-2] * self.prototype_shape[-1])**0.5

        # Normalize each 1 x 1 x latent piece to size s=64
        x_length = torch.sqrt(torch.sum(torch.square(x), dim=-3) + 1e-4)
        x_length = x_length.view(x_length.size()[0], 1, x_length.size()[1], x_length.size()[2])
        x_normalized = input_vector_length * x / x_length 
        x_normalized = x_normalized / normalizing_factor
        # x_normalized:  torch.Size([20, 128, 7, 7]) -> [batch * channel * spatial * spatial]
        # We normalize prototypes to unit length
        prototype_vector_length = torch.sqrt(torch.sum(torch.square(self.prototype_vectors), dim=-3) + 1e-4)
        prototype_vector_length = prototype_vector_length.view(prototype_vector_length.size()[0], 
                                                                1,
                                                                prototype_vector_length.size()[1],
                                                                prototype_vector_length.size()[2])
        normalized_prototypes = self.prototype_vectors / (prototype_vector_length + 1e-4)
        normalized_prototypes = normalized_prototypes / normalizing_factor
        # normalized_prototypes:  torch.Size([2000, 128, 1, 1]) -> [num_protos * channel * spatial * spatial]

        # Computing the cosine of the angle between each of the prototype/input pairs
        # Replace tensordots with conv2d by prototype
        
        distances_dot = F.conv2d(x_normalized, normalized_prototypes)

        marginless_distances = distances_dot / (input_vector_length * 1.01)
        if self.m == None or not is_train or prototypes_of_wrong_class == None:
            distances = marginless_distances
            #torch.arccos(distances_dot_div)
        # elif additive_margin:
        #     # This branch deals with additive margin
        #     prototypes_of_right_class = 1 - prototypes_of_wrong_class
        #     right_class_margin = prototypes_of_right_class * self.m
        #     right_class_margin = right_class_margin.view(x.size()[0], self.prototype_vectors.size()[0], 1, 1)
        #     right_class_margin = torch.repeat_interleave(right_class_margin, distances_dot.size()[-2], dim=-2)
        #     right_class_margin = torch.repeat_interleave(right_class_margin, distances_dot.size()[-1], dim=-1)
        #     penalized_angles = torch.arccos(distances_dot / (input_vector_length * 1.01)) + right_class_margin
        #     distances = torch.cos(torch.relu(penalized_angles))
        else:
            # This branch deals with subtractive margin
            wrong_class_margin = prototypes_of_wrong_class * self.m
            wrong_class_margin = wrong_class_margin.view(x.size()[0], self.prototype_vectors.size()[0], 1, 1)
            wrong_class_margin = torch.repeat_interleave(wrong_class_margin, distances_dot.size()[-2], dim=-2)
            wrong_class_margin = torch.repeat_interleave(wrong_class_margin, distances_dot.size()[-1], dim=-1)
            penalized_angles = torch.arccos(distances_dot / (input_vector_length * 1.01)) - wrong_class_margin
            distances = torch.cos(torch.relu(penalized_angles))

        #distances = torch.abs(distances)
        #distances:  torch.Size([20, 2000, 7, 7]) 
        #print("x_normalized: ", x_normalized.size())
        #print("normalized_prototypes: ", normalized_prototypes.size())
        #print("expanded_distances: ", expanded_distances.size())
        #print("distances: ", distances.size())
        if self.relu_on_cos:
            distances = torch.relu(distances)
            marginless_distances = torch.relu(marginless_distances)
            
        return distances, marginless_distances

    def prototype_distances(self, x, prototypes_of_wrong_class=None):
        '''
        x is the raw input
        '''
        conv_features = self.conv_features(x)
        if self.latent_space_type == 'l2':
            distances = self._l2_convolution(conv_features)
            return distances
        elif self.latent_space_type == 'arc':
            activations, marginless_activations = self.cos_activation(conv_features, prototypes_of_wrong_class=prototypes_of_wrong_class)
            return activations, marginless_activations

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def forward(self, x, prototypes_of_wrong_class=None):
        if self.latent_space_type == 'arc':
            activations, marginless_activations = self.prototype_distances(x, prototypes_of_wrong_class=prototypes_of_wrong_class)
            # global max pooling
            max_activations = F.max_pool2d(activations,
                                        kernel_size=(activations.size()[2],
                                                    activations.size()[3]))
            max_activations = max_activations.view(-1, self.num_prototypes)

            marginless_max_activations = F.max_pool2d(marginless_activations,
                                        kernel_size=(marginless_activations.size()[2],
                                                    marginless_activations.size()[3]))
            marginless_max_activations = marginless_max_activations.view(-1, self.num_prototypes)

            logits = self.last_layer(max_activations)
            marginless_logits = self.last_layer(marginless_max_activations)
            # FIXME: check where forward is called and fix min_distance -> max_activations
            return logits, marginless_logits,  max_activations
        elif self.latent_space_type == 'l2':
            distances = self.prototype_distances(x)
            '''
            we cannot refactor the lines below for similarity scores
            because we need to return min_distances
            '''
            # global min pooling
            min_distances = -F.max_pool2d(-distances,
                                        kernel_size=(distances.size()[2],
                                                    distances.size()[3]))
            min_distances = min_distances.view(-1, self.num_prototypes)
            prototype_activations = self.distance_2_similarity(min_distances)
            logits = self.last_layer(prototype_activations)
            return logits, logits, self.distance_2_similarity(min_distances)
        else:
            raise ValueError('latent_space_type not supported')


    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        conv_output = self.conv_features(x)
        if self.latent_space_type == 'l2':
            distances = self._l2_convolution(conv_output)
            activations = self.distance_2_similarity(distances)
            # FIXME: go check where push_forward is used because originally was expecting it to be min_distances
            return conv_output, activations
        else:
            _, activations = self.cos_activation(conv_output)
            # FIXME: go to push.py check where push_forward is used
            return conv_output, activations



    def prune_prototypes(self, prototypes_to_prune):
        '''
        prototypes_to_prune: a list of indices each in
        [0, current number of prototypes - 1] that indicates the prototypes to
        be removed
        '''
        prototypes_to_keep = list(set(range(self.num_prototypes)) - set(prototypes_to_prune))

        self.prototype_vectors = nn.Parameter(self.prototype_vectors.data[prototypes_to_keep, ...],
                                              requires_grad=True)

        self.prototype_shape = list(self.prototype_vectors.size())
        self.num_prototypes = self.prototype_shape[0]

        # changing self.last_layer in place
        # changing in_features and out_features make sure the numbers are consistent
        self.last_layer.in_features = self.num_prototypes
        self.last_layer.out_features = self.num_classes
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]

        # self.ones is nn.Parameter
        self.ones = nn.Parameter(self.ones.data[prototypes_to_keep, ...],
                                 requires_grad=False)
        # self.prototype_class_identity is torch tensor
        # so it does not need .data access for value update
        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]

    def __repr__(self):
        # PPNet(self, features, img_size, prototype_shape, num_classes, init_weights=True):
        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            # '\timg_size: {},\n'
            '\tprototype_shape: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                        #   self.img_size,
                          self.prototype_shape,
                          self.num_classes,
                          self.epsilon)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.last_layer_connection_weight:
            self.set_last_layer_incorrect_connection(incorrect_strength=self.last_layer_connection_weight)
        else:
            self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

    def get_prototype_orthogonalities(self):

        '''
        This method only works for prototypes with 1x1 channels
        '''
        prototype_vector_norms = F.normalize(self.prototype_vectors.squeeze(), dim=1)
        sym = torch.mm(prototype_vector_norms, torch.t(prototype_vector_norms))
        sym -= torch.eye(prototype_vector_norms.size()[0]).cuda()
        orth_loss = sym.pow(2.0).sum()
        return orth_loss


def construct_PPNet(pretrained=True,
                    proto_dim=(1, 1), singe_class_prototype_per_class=5, 
                    joint_prototypes_per_border=1, proto_depth=None, num_classes=6,
                    prototype_activation_function='log', last_layer_weight=None,
                    add_on_layers_type='regular',
                    class_specific=True, m=None, latent_space_type='l2', width=1):
    
    features = eeg_model_features(pretrained=pretrained, width=width)

    return PPNet(features=features,
                 proto_dim=proto_dim, singe_class_prototype_per_class=singe_class_prototype_per_class, 
                joint_prototypes_per_border=joint_prototypes_per_border, proto_depth=proto_depth,
                 num_classes=num_classes,
                 init_weights=True,
                 m=m,
                 prototype_activation_function=prototype_activation_function,
                 last_layer_connection_weight=last_layer_weight,
                 add_on_layers_type=add_on_layers_type,
                 class_specific=class_specific, latent_space_type=latent_space_type, width=width)

