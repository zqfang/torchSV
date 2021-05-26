

import glob,sys,os,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# model 
class CNN(nn.Module):
    def __init__(self, input_size = (256, 256), num_class=2):
        super(CNN, self).__init__()
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
        
    def forward(self, inputs):
        batch = inputs.size(0)
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