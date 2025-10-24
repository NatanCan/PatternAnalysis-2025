import torch
from torchvision import datasets as dsets
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os

class ADNIDataset(Dataset):
    def __init__(self, img_dir, b_size=32):
        self.img_dir = img_dir
        self.b_size = b_size
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def get_transforms(self):
        return self.transform
    
    def com_dsets(self, path):
        ADset = dsets.ImageFolder(root=f"{path}\\train", transform=self.get_transforms())
        NCset = dsets.ImageFolder(root=f"{path}\\train", transform=self.get_transforms())
        fullTrain = ConcatDataset((ADset, NCset))
        ADteset = dsets.ImageFolder(root=f"{path}\\test", transform=self.get_transforms())
        NCteset = dsets.ImageFolder(root=f"{path}\\test", transform=self.get_transforms())
        fullTest = ConcatDataset((ADteset, NCteset)) 
        return fullTrain, fullTest

    def load_data(self, path):
        TrainSet, TestSet = self.com_dsets(path)
        train_size = int(0.8*len(TrainSet))
        val_size = len(TrainSet) - train_size
        train_dset, val_dset = torch.utils.data.random_split(TrainSet, [train_size, val_size])
        #Creating the data Loaders (Copied from part 3)
        train_loader = DataLoader(dataset=train_dset, batch_size=self.b_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dset, batch_size=self.b_size, shuffle=False)
        test_loader = DataLoader(dataset=TestSet, batch_size=self.b_size, shuffle=False) 
        return train_loader, val_loader, test_loader
