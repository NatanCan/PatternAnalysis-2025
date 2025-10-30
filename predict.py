import torch
from torchvision import transforms
from dataset import ADNIDataset
from PIL import Image
from modules import ADNIConvNeXt

batch_size = 32
path = r"C:\Users\Nathan\Documents\AD_NC"
ADNIDset = ADNIDataset(img_dir=r"C:\Users\Nathan\Documents\AD_NC", b_size=batch_size)
_, _, test_loader = ADNIDset.load_data(path)


#Checking if GPU is available 
print("Checking for GPU...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = ADNIConvNeXt().to(device)
model.load_state_dict(torch.load("checkpoints/convnext_adni.pth", map_location="cpu"))
model.eval()

# Preprocessing
def preprocess(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)

# Predict

img = preprocess(test_loader)
outputs = model(img)
pred = torch.argmax(outputs, dim=1).item()

print(f"Prediction for {img_path}: {['Normal', 'AD'][pred]}")