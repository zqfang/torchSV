

import glob,sys,os,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# model 
class DELCNN(nn.Module):
    def __init__(self, input_size = (256, 256), num_class=2, init_weights=False):
        super(DELCNN, self).__init__()
        # padding='Same' in Keras means padding is added 
        # padding='Valid' in Keras means no padding is added.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11,11), stride=(1,1),padding=(0,0))
        self.max_pool1 = nn.MaxPool2d(kernel_size=(3,3))
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5,5))
        self.max_pool2 = nn.MaxPool2d(kernel_size=(3,3))
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3,3))
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3,3))
        self.max_pool3 = nn.MaxPool2d(kernel_size=(3,3))
        #self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3))
        self.fc1 = nn.Linear(in_features=256*7*7, out_features=512)
        self.dp1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.dp2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=512, out_features=num_class)
        if init_weights:
            self._initialize_weights()

    def forward(self, inputs):
        batch = inputs.size(0)
        # note: the order of batchnorm
        # resnet: conv -> batchnorm -> relu -> conv
        # in practice, this might work better:  conv -> relu -> bathnorm -> conv
        x = F.leaky_relu(self.conv1(inputs))
        x = self.max_pool1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.max_pool2(x)
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = self.max_pool3(x)
        # reshape to linear
        x = x.view(batch, -1)
        x = self.fc1(x)# F.leaky_relu(self.fc1(x))
        x = self.dp1(x)
        x = self.fc2(x) #F.leaky_relu(self.fc2(x))
        x = self.dp2(x)
        out = self.fc3(x)
        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# this is simpler
MODLE = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11,11), stride=(1,1),padding=(0,0)),
        # nn.BatchNorm2d(96),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=(3,3)),
        nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5,5)),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=(3,3)),
        nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3,3)),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3,3)),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=(3,3)),
        Flatten(),
        nn.Linear(in_features=256*7*7, out_features=512),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=512, out_features=512),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=512, out_features=2)
)

## the order of layers
# Enter: Convlution -> Batch Normalization -> Relu -> Max Pool
# Middle: Convlution -> Batch Normalization -> Relu
# Middle Complicated: -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
# Tail: Max Pool -> View -> (Dropout) -> Fc