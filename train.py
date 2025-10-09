#This python file defines how the model will train and validate itself using the dataset class defined in dataset.py and include the relevant parameters
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from modules import ADNIConvNeXt
from dataset import ADNIDataset, get_transforms

#Hyperparameters
Epochs = 100
batch_size = 256
learning_rate = 1e-4

#Paths for datasets
ADNIDset = ADNIDataset(img_dir=r"C:\Users\Nathan\Documents\AD_NC", transform=get_transforms())
train_loader, val_loader, test_loader = ADNIDset.load_data()

#model
"""def __init__(
        self,
        in_features: int, (1 since its grayscale)
        out_features: int, (2 since its either AD or NC)
        kernel_size: int, (3 is standard)
        norm = nn.BatchNorm2d,
        act = nn.ReLU,
        **kwargs
    ):"""

model = ADNIConvNeXt(in_features=1, out_features=2, kernel_size=3)
criteria = nn.CrossEntropyLoss()
totStep = len(train_loader)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

