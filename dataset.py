import torch
from torch.utils.data import Dataset as dsets
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
path = r"C:\Users\Nathan\Documents\AD_NC"
b_size = 256
class ADNIDataset(dsets):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_pth = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_pth).convert("RGB")
        label = torch.tensor(int(self.data.iloc[idx, 1]))

        if self.transform:
            image = self.transform(image)

        return image, label
    
    # Transforms for ConvNeXt input
    def get_transforms():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    

    def com_dsets(self):
        ADset = dsets.ImageFolder(root=f"{path}\\train\\AD", transform=self.get_transforms())
        NCset = dsets.ImageFolder(root=f"{path}\\train\\NC", transform=self.get_transforms())
        fullTrain = torch.concat((ADset, NCset), 0)
        ADteset = dsets.ImageFolder(root=f"{path}\\test\\AD", transform=self.get_transforms())
        NCteset = dsets.ImageFolder(root=f"{path}\\test\\NC", transform=self.get_transforms())
        fullTest = torch.concat((ADteset, NCteset), 0) 
        return fullTrain, fullTest

    def load_data(self, path):
        TrainSet, TestSet = self.com_dsets()
        train_size = int(0.8*len(TrainSet))
        val_size = len(TrainSet) - train_size
        train_dset, val_dset = torch.utils.data.random_split(TrainSet, [train_size, val_size])
        #Creating the data Loaders (Copied from part 3)
        train_loader = DataLoader(dataset=train_dset, batch_size=b_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dset, batch_size=b_size, shuffle=False)
        test_loader = DataLoader(dataset=TestSet, batch_size=b_size, shuffle=False) 
        return train_loader, val_loader, test_loader
