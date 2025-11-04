import os
import torch
from torchvision import models, transforms
from PIL import Image

# =========================
# Configuration
# =========================
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/agrisense_resnet18.pth"))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/train"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names from training directory
class_names = sorted(os.listdir(DATA_DIR))
print(f"✅ Loaded {len(class_names)} classes.")

# =========================
# Load Model
# =========================
model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)
print("✅ Model loaded successfully!")

# =========================
# Transform
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# Prediction Function
# =========================
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    print(f"🌿 Predicted Disease: {predicted_class}")

# =========================
# Run Prediction
# =========================
if __name__ == "__main__":
    test_image_path = input("🖼️ Enter path of image to predict: ").strip('"')
    if os.path.exists(test_image_path):
        predict_image(test_image_path)
    else:
        print("❌ Invalid path. Please check and try again.")
