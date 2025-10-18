#This python file defines how the model will train and validate itself using the dataset class defined in dataset.py and include the relevant parameters
import torch
import torch.nn as nn
import os
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

#storing loss values and validation accuracy
train_loss, val_loss, val_accs = [],[],[]

for epoch in range(Epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criteria(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / totStep
    train_loss.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{Epochs}], Training Loss: {epoch_loss:.4f}")

    #Validation phase
    model.eval()
    val_running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criteria(outputs, labels)
            val_running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_epoch_loss = val_running_loss / len(val_loader)
    val_loss.append(val_epoch_loss)
    val_acc = accuracy_score(all_labels, all_preds)
    val_accs.append(val_acc)
    print(f"Epoch [{epoch+1}/{Epochs}], Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    
