import torch
from torchvision import transforms
import torch.nn as nn
from dataset import ADNIDataset
from PIL import Image
from modules import ADNIConvNeXt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Set up the test data loader for model evaluation.
batch_size = 256
path = r"C:\Users\Nathan\Documents\AD_NC"
ADNIDset = ADNIDataset(img_dir=r"C:\Users\Nathan\Documents\AD_NC", b_size=batch_size)
_, _, test_loader = ADNIDset.load_data(path)


#Checking if GPU is available 
print("Checking for GPU...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model of the trained model to test accuracy
model = ADNIConvNeXt().to(device)
model.load_state_dict(torch.load(
    r"C:\Users\Nathan\Documents\COMP3710\COMP3710ReportCode\checkpoints\convnext_adni.pth",
    map_location=device
))
model.eval()

# === Evaluation ===
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# === Metrics ===
acc = accuracy_score(all_labels, all_preds)
print(f"\n Test Accuracy: {acc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["Normal", "AD"]))

