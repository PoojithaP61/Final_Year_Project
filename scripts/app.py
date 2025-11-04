import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os
import json
import numpy as np
import cv2

# ======================
# Configuration
# ======================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
MODEL_PATH = os.path.join(ROOT, "models", "agrisense_resnet18.pth")
DATA_DIR = os.path.join(ROOT, "data", "train")
INFO_PATH = os.path.join(ROOT, "docs", "disease_treatments.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Load class names
# ======================
class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

# ======================
# Load disease info JSON
# ======================
if os.path.exists(INFO_PATH):
    with open(INFO_PATH, "r", encoding="utf-8") as f:
        disease_info = json.load(f)
else:
    disease_info = {}

# ======================
# Load model
# ======================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval().to(DEVICE)
    return model

model = load_model()

# ======================
# Image transformation
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ======================
# Grad-CAM utilities
# ======================
def generate_gradcam(model, img_tensor, target_class):
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, inp, out):
        activations.append(out)

    last_conv = model.layer4[-1].conv2
    h1 = last_conv.register_forward_hook(forward_hook)
    h2 = last_conv.register_backward_hook(backward_hook)

    output = model(img_tensor)
    model.zero_grad()
    class_loss = output[0, target_class]
    class_loss.backward()

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam) if np.max(cam) != 0 else cam

    h1.remove()
    h2.remove()
    return cam

def overlay_heatmap(image_pil, heatmap):
    img = np.array(image_pil)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(overlay)

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="AgriSense+", page_icon="🌿")
st.title("🌿 AgriSense+ — Plant Disease Detection")
st.write("Upload a leaf image to detect the disease, view heatmap and treatment details.")

uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        conf, predicted = torch.max(probs, 0)
        pred_class = class_names[predicted.item()]

    st.success(f"🌱 Predicted Disease: **{pred_class}**  |  Confidence: {conf.item() * 100:.2f}%")

    # Grad-CAM heatmap
    try:
        heatmap = generate_gradcam(model, img_tensor, predicted.item())
        overlay = overlay_heatmap(image.resize((224, 224)), heatmap)
        st.image(overlay, caption="🔍 Grad-CAM Visualization (diseased area highlight)", use_column_width=True)
    except Exception as e:
        st.warning(f"Grad-CAM visualization unavailable: {e}")

    # Disease Info
    st.markdown("---")
    st.markdown("### 🩺 Disease Information")

    info = disease_info.get(pred_class)
    if info:
        st.write(f"**Cause:** {info.get('cause','-')}")
        st.write(f"**Symptoms:** {info.get('symptoms','-')}")
        st.write(f"**Treatment:** {info.get('treatment','-')}")
    else:
        st.info("No detailed info found for this disease in docs/disease_treatments.json.")

st.markdown("---")
st.caption("Model: ResNet-18 trained on PlantVillage | Accuracy: 98.38% | Powered by PyTorch + Streamlit 🌾")
