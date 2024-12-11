import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchsummary import summary

class _DenseLayer(nn.Sequential):
    """
    A single dense layer in the DenseNet architecture.

    Args:
        num_input_features (int): Number of input features.
        growth_rate (int): Growth rate for the layer.
        bn_size (int): Bottleneck size.
        drop_rate (float): Dropout rate.
        conv_bias (bool): Use bias in convolution layers.
        batch_norm (bool): Apply batch normalization.
    """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, conv_bias, batch_norm):
        super(_DenseLayer, self).__init__()
        if batch_norm:
            self.add_module('norm1', nn.BatchNorm1d(num_input_features))
        self.add_module('elu1', nn.ELU())
        self.add_module('conv1', nn.Conv1d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=conv_bias))
        if batch_norm:
            self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate))
        self.add_module('elu2', nn.ELU())
        self.add_module('conv2', nn.Conv1d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=conv_bias))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    """
    A block of dense layers in DenseNet.

    Args:
        num_layers (int): Number of dense layers.
        num_input_features (int): Number of input features.
        bn_size (int): Bottleneck size.
        growth_rate (int): Growth rate.
        drop_rate (float): Dropout rate.
        conv_bias (bool): Use bias in convolution layers.
        batch_norm (bool): Apply batch normalization.
    """
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, conv_bias, batch_norm):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, conv_bias, batch_norm)
            self.add_module(f'denselayer{i + 1}', layer)

class _Transition(nn.Sequential):
    """
    Transition layer between DenseNet blocks to reduce feature maps.

    Args:
        num_input_features (int): Number of input features.
        num_output_features (int): Number of output features.
        conv_bias (bool): Use bias in convolution layers.
        batch_norm (bool): Apply batch normalization.
    """
    def __init__(self, num_input_features, num_output_features, conv_bias, batch_norm):
        super(_Transition, self).__init__()
        if batch_norm:
            self.add_module('norm', nn.BatchNorm1d(num_input_features))
        self.add_module('elu', nn.ELU())
        self.add_module('conv', nn.Conv1d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=conv_bias))
        self.add_module('pool', nn.AvgPool1d(kernel_size=2, stride=2))

class DenseNetEncoder(nn.Module):
    """
    DenseNet Encoder for feature extraction.

    Args:
        growth_rate (int): Growth rate.
        block_config (tuple): Configuration of layers in each block.
        in_channels (int): Number of input channels.
        num_init_features (int): Number of initial features.
        bn_size (int): Bottleneck size.
        drop_rate (float): Dropout rate.
        conv_bias (bool): Use bias in convolution layers.
        batch_norm (bool): Apply batch normalization.
    """
    def __init__(self, growth_rate=32, block_config=(4, 4, 4, 4, 4, 4, 4), in_channels=16, num_init_features=64, 
                 bn_size=4, drop_rate=0.2, conv_bias=True, batch_norm=False):
        super(DenseNetEncoder, self).__init__()
        # Initial convolution
        first_conv = OrderedDict([
            ('conv0', nn.Conv1d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=conv_bias))
        ])
        if batch_norm:
            first_conv['norm0'] = nn.BatchNorm1d(num_init_features)
        first_conv['elu0'] = nn.ELU()
        first_conv['pool0'] = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.densenet = nn.Sequential(first_conv)
        num_features = num_init_features

        # Dense blocks and transitions
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate, conv_bias, batch_norm)
            self.densenet.add_module(f'denseblock{i + 1}', block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features // 2, conv_bias, batch_norm)
                self.densenet.add_module(f'transition{i + 1}', trans)
                num_features //= 2

        # Final layers
        if batch_norm:
            self.densenet.add_module(f'norm{len(block_config) + 1}', nn.BatchNorm1d(num_features))
        self.densenet.add_module(f'relu{len(block_config) + 1}', nn.ReLU())
        self.densenet.add_module(f'pool{len(block_config) + 1}', nn.AvgPool1d(kernel_size=7, stride=3))
        self.num_features = num_features

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.densenet(x).view(x.size(0), -1)

class DenseNetClassifier(nn.Module):
    """
    DenseNet Classifier for EEG data.

    Args:
        growth_rate (int): Growth rate.
        block_config (tuple): Configuration of layers in each block.
        in_channels (int): Number of input channels.
        num_init_features (int): Number of initial features.
        bn_size (int): Bottleneck size.
        drop_rate (float): Dropout rate.
        conv_bias (bool): Use bias in convolution layers.
        batch_norm (bool): Apply batch normalization.
        drop_fc (float): Dropout rate for fully connected layer.
        num_classes (int): Number of output classes.
        width (int): Number of feature windows.
    """
    def __init__(self, growth_rate=32, block_config=(4, 4, 4, 4, 4, 4, 4), in_channels=16, num_init_features=64,
                 bn_size=4, drop_rate=0.2, conv_bias=True, batch_norm=False, drop_fc=0.5, num_classes=6, width=1):
        super(DenseNetClassifier, self).__init__()
        self.features = DenseNetEncoder(growth_rate, block_config, in_channels, num_init_features, bn_size, drop_rate, conv_bias, batch_norm)
        self.width = width

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.width == 1:
            features = self.features(x[:, :, 4000:6000])
        elif self.width == 3:
            feature1 = self.features(x[:, :, 2000:4000])
            feature2 = self.features(x[:, :, 4000:6000])
            feature3 = self.features(x[:, :, 6000:8000])
            features = torch.cat([feature1, feature2, feature3], dim=1)
        elif self.width == 5:
            feature1 = self.features(x[:, :, 0:2000])
            feature2 = self.features(x[:, :, 2000:4000])
            feature3 = self.features(x[:, :, 4000:6000])
            feature4 = self.features(x[:, :, 6000:8000])
            feature5 = self.features(x[:, :, 8000:10000])
            features = torch.cat([feature1, feature2, feature3, feature4, feature5], dim=1)

        return features.view(features.size(0), features.size(1), 1, 1)

def eeg_model_features(pretrained=False, width=1):
    """
    Instantiate EEG DenseNet model with optional pretrained weights.

    Args:
        pretrained (bool): Load pretrained weights.
        width (int): Number of feature windows.

    Returns:
        DenseNetClassifier: Initialized model.
    """
    model = DenseNetClassifier(width=width)
    if pretrained:
        state_dict = torch.load("/usr/xtmp/zg78/proto_proj/demo/model_1130.pt").state_dict()
        del state_dict["classifier.1.weight"], state_dict["classifier.1.bias"]
        model.load_state_dict(state_dict)
    return model

if __name__ == '__main__':
    model = eeg_model_features(pretrained=True)
    print(summary(model, (16, 2000)))
