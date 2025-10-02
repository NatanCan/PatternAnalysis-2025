import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class ADNIDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_pth = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
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