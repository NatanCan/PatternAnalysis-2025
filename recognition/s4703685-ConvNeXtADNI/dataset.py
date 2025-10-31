import torch
from torchvision import datasets as dsets
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

"""
    A dataset class for loading and preprocessing ADNI (Alzheimer's Disease Neuroimaging Initiative) data.
    
    This class handles the loading of brain MRI images from a structured directory, applies appropriate
    transformations for training and testing, and creates DataLoader objects for model training and evaluation.
    
    Attributes:
        img_dir (str): Directory path containing the image data
        b_size (int): Batch size for DataLoader objects
        train_transform (transforms.Compose): Composition of transformations for training data
        test_transform (transforms.Compose): Composition of transformations for test/validation data
"""
class ADNIDataset:
    
    """
        Initialize the ADNIDataset with directory path and batch size.
        
        Args:
            img_dir (str): Path to the directory containing the image data
            b_size (int, optional): Batch size for DataLoader. Defaults to 128.
            
        Note:
            The directory structure should be organized as:
            - img_dir/train/AD/
            - img_dir/train/Normal/
            - img_dir/test/AD/
            - img_dir/test/Normal/
    """
    def __init__(self, img_dir, b_size=128):
        self.img_dir = img_dir
        self.b_size = b_size
        
        # Training transformations with data augmentation
        self.train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

        # Test/validation transformations (minimal preprocessing)
        self.test_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    """
        Get the appropriate transformation pipeline based on the data split.
        
        Args:
            train (bool): If True, returns training transformations with data augmentation.
                         If False, returns test/validation transformations.
        
        Returns:
            transforms.Compose: The composed transformation pipeline
    """
    def get_transforms(self, train):
        if train == True:
            return self.train_transform
        else:
            return self.test_transform
        
    """
        Load and split the dataset into train, validation, and test DataLoaders.
        
        Args:
            path (str): Path to the main directory containing 'train' and 'test' subdirectories
        
        Returns:
            tuple: A tuple containing:
                - train_loader (DataLoader): DataLoader for training data
                - val_loader (DataLoader): DataLoader for validation data  
                - test_loader (DataLoader): DataLoader for test data
    """
    def load_data(self, path):
        # Folders should be structured like:
        # path/train/AD, path/train/Normal
        # path/test/AD, path/test/Normal
        train_dataset = dsets.ImageFolder(root=f"{path}\\train", transform=self.get_transforms(True))
        test_dataset = dsets.ImageFolder(root=f"{path}\\test", transform=self.get_transforms(False))

        # 80/20 train/val split (no duplication)
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dset, val_dset = random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dset, batch_size=self.b_size, shuffle=True)
        val_loader = DataLoader(val_dset, batch_size=self.b_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.b_size, shuffle=False)
        return train_loader, val_loader, test_loader
