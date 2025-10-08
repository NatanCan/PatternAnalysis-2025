#This python file defines how the model will train and validate itself using the dataset class defined in dataset.py and include the relevant parameters
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from modules import ADNIConvNeXt
from dataset import ADNIDataset, get_transforms


#Hyperparameters
Epochs = 100
batch_size = 256
lr = 1e-4

#Paths for datasets
#TBI

