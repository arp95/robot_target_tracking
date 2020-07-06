# header files
import os
import torch
import torchvision
import torch.nn as nn
import torch.autograd.functional as F
import numpy as np
import random


def _init_weights(module: nn.Module):
    if type(module) == nn.Linear:
        torch.nn.init.xavier_uniform(module.weight)
        module.bias.data.fill_(0.01)

# neural net class
class ResNet34(torch.nn.Module):
    
    # init function
    def __init__(self, num_classes=512):
        super(ResNet34, self).__init__()
        
        self.features = torchvision.models.resnet34(pretrained=True)
        self.features.conv1 = nn.Conv2d(1, 64, kernel_size=7, padding=3, stride=2, bias=False)
        num_features = self.features.fc.out_features
        
        self.features.apply(_init_weights)
        self.fc1 = nn.Linear(num_features, num_features)
        self.fc2 = nn.Linear(num_features, num_features)
        self.fc3 = nn.Linear(num_features, num_classes)
        
    # forward function
    def forward(self, x):
        x = F.relu(self.features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
