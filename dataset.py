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
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def com_dsets(self, path):
        ADset = dsets.ImageFolder(root=f"{path}\\train\\AD", transform=self.get_transforms())
        NCset = dsets.ImageFolder(root=f"{path}\\train\\NC", transform=self.get_transforms())
        fullTrain = ConcatDataset((ADset, NCset), 0)
        ADteset = dsets.ImageFolder(root=f"{path}\\test\\AD", transform=self.get_transforms())
        NCteset = dsets.ImageFolder(root=f"{path}\\test\\NC", transform=self.get_transforms())
        fullTest = ConcatDataset((ADteset, NCteset), 0) 
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
