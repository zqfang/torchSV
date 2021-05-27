

import glob, os, sys
import pandas as pd
import skimage
import torch
from torch.utils.data.dataset import Dataset

# from torchvision import transforms

# tmfs = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

class DelDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, transform=None):
        """
            csv_file (string): Path to the csv file with annotations (label).
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.deletions_frame = pd.read_csv(csv_file) # col: image_name, label
        self.transform = transform
        self.root_dir = root_dir
        
    def __len__(self):
        return len(self.deletions_frame)  
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.deletions_frame.iloc[idx, 0])
        image = skimage.io.imread(img_name)
        label = self.landmarks_frame.iloc[idx, 1]
        # label = label.astype('float') # 
        label = torch.from_numpy(label).type(torch.LongTensor)
        image = image.transpose((2, 0, 1))
        sample = {'image': torch.from_numpy(image), 'label': label}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def toTensor(self, sample: dict) -> dict[str, torch.Tensor]:
        image, label = sample['image'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}    
        # torch.LongTensor