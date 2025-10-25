import torch
from torchvision import transforms
from PIL import Image
from modules import ADNIConvNeXt

# Load model
model = ADNIConvNeXt()
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
img_path = "test_image.jpg"
img = preprocess(img_path)
outputs = model(img)
pred = torch.argmax(outputs, dim=1).item()

print(f"Prediction for {img_path}: {['Normal', 'AD'][pred]}")