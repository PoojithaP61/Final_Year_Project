import torch
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
ENCODER_PATH = os.path.join(ROOT, "models", "few_shot_encoder.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained encoder
encoder = models.mobilenet_v3_small(weights=None)
encoder.classifier = torch.nn.Identity()
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
encoder = encoder.to(DEVICE).eval()

# Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = encoder(tensor)
    return emb.squeeze().cpu().numpy()

# Few-shot classification using cosine similarity
def predict_few_shot(test_img, support_images, support_labels):
    test_emb = get_embedding(test_img)
    support_embs = np.stack([get_embedding(p) for p in support_images])
    sims = np.dot(support_embs, test_emb) / (
        np.linalg.norm(support_embs, axis=1) * np.linalg.norm(test_emb)
    )
    return support_labels[np.argmax(sims)]
