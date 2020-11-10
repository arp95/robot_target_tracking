# reference: https://github.com/ksengin/active-target-localization/blob/master/target_localization/models/convnet.py
import os

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision


def _init_weights(module: nn.Module):
    if type(module) == nn.Linear:
        torch.nn.init.xavier_uniform(module.weight)
        module.bias.data.fill_(0.01)


class ConvNet(nn.Module):
    """
    The ConvNet class defines a convolutional neural net with a desired output dimension.
    """

    def __init__(self, out_dim: int = 128, pretrained=True):
        """
        Create a convolutional network.
        """
        super(ConvNet, self).__init__()

        self.act_fn = F.relu
        self.model_ft = torchvision.models.resnet34(pretrained=pretrained)
        self.model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model_ft.fc.out_features

        self.model_ft.apply(_init_weights)
        self.fc1 = nn.Linear(num_ftrs, num_ftrs)
        self.fc2 = nn.Linear(num_ftrs, num_ftrs)
        self.fc3 = nn.Linear(num_ftrs, out_dim)


    def forward(self, x: torch.tensor):
        x = self.act_fn(self.model_ft(x))
        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x))
        fc_out = self.fc3(x)
        return fc_out
