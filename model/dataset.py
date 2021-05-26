

import glob, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import average_precision_score
# from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, accuracy_score

## TODO
class DelDataset(Dataset):
    def __init__(self, inputs, targets):
        inputs = inputs.astype(np.float32)
        targets = targets.astype(np.float32)
        self.inputs = torch.from_numpy(inputs)
        self.targets = torch.from_numpy(targets)

    def __getitem__(self, idx):
        ## multi-class classification need longTensor for y
        self.targets = self.targets.type(torch.LongTensor)
        return {'embed': self.inputs[idx], 'target': self.targets[idx]}
    def __len__(self):
        return len(self.targets)  