import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader
from glob import glob
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self,lr_dir,hr_dir,transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform
        
        self.lr_filepath = sorted(glob(os.path.join(self.lr_dir,'*.npy')))
        self.hr_filepath = sorted(glob(os.path.join(self.hr_dir,'*.npy')))
        
    def __len__(self):
        return len(self.hr_filepath)
    
    def __getitem__(self,index):
        lr = np.load(self.lr_filepath[index])
        hr = np.load(self.hr_filepath[index])
        
        if self.transform:
            lr = self.transform(lr)
            hr = self.transform(hr)
        
        return lr , hr


dataset = CustomDataset(lr_dir='/content/LR',hr_dir='/content/HR', transform = transforms.Compose([transforms.ToTensor()]))

train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset)*.80), int(len(dataset)*.20)])

train_loader = DataLoader(train_set,batch_size=4,shuffle=True)
test_loader = DataLoader(test_set,batch_size=4,shuffle=True)        