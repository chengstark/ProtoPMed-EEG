import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from itertools import combinations
from eeg_model_features import eeg_model_features

class PPNet(nn.Module):
    """
    Prototype-based Neural Network (PPNet).

    Attributes:
        features (nn.Module): Feature extraction module.
        proto_dim (tuple): Prototype dimensions.
        singe_class_prototype_per_class (int): Number of prototypes per class.
        joint_prototypes_per_border (int): Number of joint prototypes across class borders.
        proto_depth (int): Depth of prototypes.
        num_classes (int): Number of classes.
        init_weights (bool): Whether to initialize weights.
        last_layer_connection_weight (float): Weight for the last layer connection.
        prototype_activation_function (str): Prototype activation function.
        m (float): Margin value for cosine similarity activation.
        add_on_layers_type (str): Type of add-on layers.
        latent_space_type (str): Latent space type ('l2' or 'arc').
        class_specific (bool): Whether prototypes are class-specific.
        width (int): Width multiplier for feature extraction.
    """

    def __init__(
        self,
        features,
        proto_dim=(1, 1),
        singe_class_prototype_per_class=5,
        joint_prototypes_per_border=1,
        proto_depth=None,
        num_classes=3,
        init_weights=True,
        last_layer_connection_weight=None,
        prototype_activation_function='log',
        m=None,
        add_on_layers_type='bottleneck',
        latent_space_type='l2',
        class_specific=False,
        width=1
    ):
        super(PPNet, self).__init__()

        self.joint_prototypes_per_border = joint_prototypes_per_border
        self.singe_class_prototype_per_class = singe_class_prototype_per_class
        self.num_classes = num_classes
        self.num_prototypes = (
            self.singe_class_prototype_per_class * self.num_classes +
            self.joint_prototypes_per_border * self.ncr(self.num_classes, 2)
        )
        self.prototype_shape = (
            self.num_prototypes, proto_depth, proto_dim[0], proto_dim[1]
        )
        self.topk_k = 1
        self.class_specific = class_specific
        self.epsilon = 1e-4
        self.m = m
        self.relu_on_cos = False
        self.input_vector_length = 64
        self.last_layer_connection_weight = last_layer_connection_weight
        self.latent_space_type = latent_space_type
        self.prototype_activation_function = prototype_activation_function

        # Initialize prototype class identity matrix
        self.prototype_class_identity = torch.zeros(self.num_prototypes, self.num_classes)
        for j in range(self.singe_class_prototype_per_class * self.num_classes):
            self.prototype_class_identity[j, j // self.singe_class_prototype_per_class] = 1

        col_index = list(combinations(range(self.num_classes), 2))
        for idx, row_index in enumerate(
            range(self.singe_class_prototype_per_class * self.num_classes, self.num_prototypes)
        ):
            self.prototype_class_identity[row_index, col_index[idx % len(col_index)][0]] = 1
            self.prototype_class_identity[row_index, col_index[idx % len(col_index)][1]] = 1

        self.features = features
        first_add_on_layer_in_channels = 255 * width

        # Initialize add-on layers
        if add_on_layers_type == 'bottleneck':
            self.add_on_layers = self._create_bottleneck_layers(first_add_on_layer_in_channels)
        elif add_on_layers_type == 'identity':
            self.add_on_layers = nn.Sequential(nn.Identity())
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
            )

        # Prototype vectors
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        # Last layer
        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)

        if init_weights:
            self._initialize_weights()

    def _create_bottleneck_layers(self, first_add_on_layer_in_channels):
        """Creates bottleneck layers."""
        add_on_layers = []
        current_in_channels = first_add_on_layer_in_channels
        while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
            current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
            add_on_layers.append(
                nn.Conv2d(in_channels=current_in_channels, out_channels=current_out_channels, kernel_size=1)
            )
            add_on_layers.append(nn.ReLU())
            add_on_layers.append(
                nn.Conv2d(in_channels=current_out_channels, out_channels=current_out_channels, kernel_size=1)
            )
            if current_out_channels > self.prototype_shape[1]:
                add_on_layers.append(nn.ReLU())
            else:
                assert current_out_channels == self.prototype_shape[1]
                add_on_layers.append(nn.Sigmoid())
            current_in_channels = current_in_channels // 2
        return nn.Sequential(*add_on_layers)

    def ncr(self, n, r):
        """Calculates combinations (n choose r)."""
        r = min(r, n - r)
        numer = reduce(lambda x, y: x * y, range(n, n - r, -1), 1)
        denom = reduce(lambda x, y: x * y, range(1, r + 1), 1)
        return numer // denom

    def forward(self, x, prototypes_of_wrong_class=None):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            prototypes_of_wrong_class (torch.Tensor): Prototypes of wrong classes.

        Returns:
            Tuple[torch.Tensor, ...]: Model predictions and prototype activations.
        """
        if self.latent_space_type == 'arc':
            activations, marginless_activations = self.prototype_distances(x, prototypes_of_wrong_class)
            max_activations = F.max_pool2d(activations, kernel_size=(activations.size()[2], activations.size()[3]))
            max_activations = max_activations.view(-1, self.num_prototypes)

            marginless_max_activations = F.max_pool2d(
                marginless_activations, kernel_size=(marginless_activations.size()[2], marginless_activations.size()[3])
            )
            marginless_max_activations = marginless_max_activations.view(-1, self.num_prototypes)

            logits = self.last_layer(max_activations)
            marginless_logits = self.last_layer(marginless_max_activations)
            return logits, marginless_logits, max_activations

        elif self.latent_space_type == 'l2':
            distances = self.prototype_distances(x)
            min_distances = -F.max_pool2d(-distances, kernel_size=(distances.size()[2], distances.size()[3]))
            min_distances = min_distances.view(-1, self.num_prototypes)
            prototype_activations = self.distance_2_similarity(min_distances)
            logits = self.last_layer(prototype_activations)
            return logits, logits, prototype_activations

        else:
            raise ValueError('latent_space_type not supported')

    def prototype_distances(self, x, prototypes_of_wrong_class=None):
        """Calculates distances between prototypes and input x."""
        conv_features = self.conv_features(x)
        if self.latent_space_type == 'l2':
            distances = self._l2_convolution(conv_features)
            return distances
        elif self.latent_space_type == 'arc':
            activations, marginless_activations = self.cos_activation(conv_features, prototypes_of_wrong_class)
            return activations, marginless_activations

    def push_forward(self, x):
        """Performs the pushing operation forward pass."""
        conv_output = self.conv_features(x)
        if self.latent_space_type == 'l2':
            distances = self._l2_convolution(conv_output)
            activations = self.distance_2_similarity(distances)
            return conv_output, activations
        else:
            _, activations = self.cos_activation(conv_output)
            return conv_output, activations

    def conv_features(self, x):
        """Extracts convolutional features."""
        x = self.features(x)
        x = self.add_on_layers(x)
        return x

    def _initialize_weights(self):
        """Initializes weights."""
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.last_layer_connection_weight:
            self.set_last_layer_incorrect_connection(self.last_layer_connection_weight)
        else:
            self.set_last_layer_incorrect_connection(-0.5)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        """Sets incorrect connection strength for the last layer."""
        positive_weights = torch.t(self.prototype_class_identity)
        negative_weights = 1 - positive_weights

        correct_connection = 1
        incorrect_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_connection * positive_weights + incorrect_connection * negative_weights
        )

    def _l2_convolution(self, x):
        """Performs L2 convolution between input and prototypes."""
        distances = F.conv2d(
            input=x ** 2,
            weight=-2 * self.prototype_vectors,
            bias=self.prototype_vectors.view(self.num_prototypes, -1).sum(dim=1),
        )
        return distances

    def distance_2_similarity(self, distances):
        """Converts distances to similarity scores."""
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def prune_prototypes(self, prototypes_to_prune):
        """Prunes specified prototypes."""
        prototypes_to_keep = list(set(range(self.num_prototypes)) - set(prototypes_to_prune))

        self.prototype_vectors = nn.Parameter(self.prototype_vectors.data[prototypes_to_keep], requires_grad=True)
        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep]
        self.num_prototypes = len(prototypes_to_keep)
        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)

        self.ones = nn.Parameter(self.ones.data[prototypes_to_keep], requires_grad=False)

    def __repr__(self):
        """Returns string representation of the model."""
        return (f"PPNet(\n"
                f"  features={self.features},\n"
                f"  prototype_shape={self.prototype_shape},\n"
                f"  num_classes={self.num_classes},\n"
                f"  epsilon={self.epsilon}\n)")


def construct_PPNet(pretrained=True, proto_dim=(1, 1), singe_class_prototype_per_class=5,
                    joint_prototypes_per_border=1, proto_depth=None, num_classes=6,
                    prototype_activation_function='log', last_layer_weight=None,
                    add_on_layers_type='regular', class_specific=True, m=None,
                    latent_space_type='l2', width=1):
    """Constructs a PPNet model."""
    features = eeg_model_features(pretrained=pretrained, width=width)
    return PPNet(
        features=features,
        proto_dim=proto_dim,
        singe_class_prototype_per_class=singe_class_prototype_per_class,
        joint_prototypes_per_border=joint_prototypes_per_border,
        proto_depth=proto_depth,
        num_classes=num_classes,
        init_weights=True,
        m=m,
        prototype_activation_function=prototype_activation_function,
        last_layer_connection_weight=last_layer_weight,
        add_on_layers_type=add_on_layers_type,
        class_specific=class_specific,
        latent_space_type=latent_space_type,
        width=width,
    )
