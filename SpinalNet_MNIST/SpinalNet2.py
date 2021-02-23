from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

model_ft = models.wide_resnet101_2(pretrained=True)
num_ftrs = model_ft.fc.in_features

half_in_size = round(num_ftrs / 2)
layer_width = 20  # Small for Resnet, large for VGG
Num_class = 10


class SpinalNet_ResNet(nn.Module):
    def __init__(self):
        super(SpinalNet_ResNet, self).__init__()

        self.fc_spinal_layer1 = nn.Sequential(
            # nn.Dropout(p = 0.5),
            nn.Linear(half_in_size, layer_width),
            # nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True), )
        self.fc_spinal_layer2 = nn.Sequential(
            # nn.Dropout(p = 0.5),
            nn.Linear(half_in_size + layer_width, layer_width),
            # nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True), )
        self.fc_spinal_layer3 = nn.Sequential(
            # nn.Dropout(p = 0.5),
            nn.Linear(half_in_size + layer_width, layer_width),
            # nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True), )
        self.fc_spinal_layer4 = nn.Sequential(
            # nn.Dropout(p = 0.5),
            nn.Linear(half_in_size + layer_width, layer_width),
            # nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True), )
        self.fc_out = nn.Sequential(
            # nn.Dropout(p = 0.5),
            nn.Linear(layer_width * 4, Num_class), )

    def forward(self, x):
        x1 = self.fc_spinal_layer1(x[:, 0:half_in_size])
        x2 = self.fc_spinal_layer2(torch.cat([x[:, half_in_size:2 * half_in_size], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([x[:, 0:half_in_size], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([x[:, half_in_size:2 * half_in_size], x3], dim=1))

        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)

        x = self.fc_out(x)
        return x




