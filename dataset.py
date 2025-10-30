import torch
from torchvision import datasets as dsets
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

class ADNIDataset:
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
    
    def load_data(self, path):
        # Folders should be structured like:
        # path/train/AD, path/train/Normal
        # path/test/AD, path/test/Normal
        train_dataset = dsets.ImageFolder(root=f"{path}\\train", transform=self.get_transforms())
        test_dataset = dsets.ImageFolder(root=f"{path}\\test", transform=self.get_transforms())

        # 80/20 train/val split (no duplication)
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dset, val_dset = random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dset, batch_size=self.b_size, shuffle=True)
        val_loader = DataLoader(val_dset, batch_size=self.b_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.b_size, shuffle=False)
        return train_loader, val_loader, test_loader
