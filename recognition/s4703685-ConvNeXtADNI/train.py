#This python file defines how the model will train and validate itself using the dataset class defined in dataset.py and include the relevant parameters
import torch
import torch.nn as nn
import os
import time
import torch.optim as optim
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from modules import ADNIConvNeXt
from dataset import ADNIDataset

#Hyperparameters
Epochs = 60
batch_size = 128
learning_rate = 3e-5

#Checking if GPU is available 
print("Checking for GPU...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Paths for datasets
print("Loading Dataset...")
path = r"C:\Users\Nathan\Documents\AD_NC"
ADNIDset = ADNIDataset(img_dir=r"C:\Users\Nathan\Documents\AD_NC", b_size=batch_size)
train_loader, val_loader, test_loader = ADNIDset.load_data(path)

#model description
"""def __init__(
        self,
        in_features: int, (3 RGB inputs)
        out_features: int, (2 since its either AD or NC)
        kernel_size: int, (3 is standard)
        norm = nn.BatchNorm2d,
        act = nn.ReLU,
        **kwargs
    ):"""
print("Initializing Model...")

#Initialize the ADNI ConvNeXt model, loss function, and optimizer.
model = ADNIConvNeXt(in_features=3, out_features=2).to(device)
criteria = nn.CrossEntropyLoss()
totStep = len(train_loader)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.05)

#storing loss values and validation accuracy
train_loss, val_loss, val_accs = [],[],[]
print("Starting Training...")

#Model training and validation loop
for epoch in range(Epochs):
    start = time.time() #time generation
    
    #Training phase
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criteria(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    if i % 10 == 0:
        print(f"Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")
    epoch_loss = running_loss / totStep
    train_loss.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{Epochs}], Training Loss: {epoch_loss:.4f}")
    end = time.time()
    print(f"Training took {(end-start)/60:.1f} minutes")
    
    #Validation phase
    model.eval()
    val_running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
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
        
# Save model
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/convnext_adni.pth")

# === Plot loss curves ===
plt.figure(figsize=(10, 5))
plt.plot(train_loss, label='Train Loss', color='blue', linewidth=2)
plt.plot(val_loss, label='Validation Loss', color='orange', linewidth=2)
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()

# === Plot accuracy curve ===
plt.figure(figsize=(10, 5))
plt.plot(val_accs, label='Validation Accuracy', color='green', linewidth=2)
plt.title("Validation Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("val_accuracy_curve.png")
plt.show()