import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, datasets, transforms
from sklearn.manifold import TSNE
from tqdm import tqdm

# ======================
# Config
# ======================
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/train"))
ENCODER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/few_shot_encoder.pth"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Data transform
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ======================
# Load dataset
# ======================
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
class_names = dataset.classes
print(f"✅ Loaded {len(dataset)} images from {len(class_names)} classes.")

# ======================
# Load encoder
# ======================
encoder = models.mobilenet_v3_small(weights=None)
encoder.classifier = torch.nn.Identity()
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
encoder = encoder.to(DEVICE).eval()
print("✅ Encoder loaded successfully!")

# ======================
# Extract embeddings
# ======================
embeddings, labels = [], []
with torch.no_grad():
    for imgs, lbls in tqdm(loader, desc="Extracting embeddings"):
        imgs = imgs.to(DEVICE)
        feats = encoder(imgs)
        feats = torch.flatten(feats, 1)  # flatten features
        embeddings.append(feats.cpu().numpy())
        labels.append(lbls.numpy())

embeddings = np.concatenate(embeddings)
labels = np.concatenate(labels)

print(f"Embeddings shape: {embeddings.shape}, Labels shape: {labels.shape}")

# ======================
# t-SNE Visualization
# ======================
print("Running t-SNE (this may take a few minutes)...")
tsne = TSNE(n_components=2, init='random', perplexity=30, random_state=42)
emb_2d = tsne.fit_transform(embeddings)

# ======================
# Plot
# ======================
plt.figure(figsize=(10, 8))
for i, cls in enumerate(class_names):
    idxs = np.where(labels == i)
    plt.scatter(emb_2d[idxs, 0], emb_2d[idxs, 1], label=cls, s=15)

plt.title("t-SNE Visualization of Contrastive Encoder Embeddings", fontsize=14)
plt.legend(fontsize=8, loc="best", ncol=2)
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.tight_layout()

# Save the plot
SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/tsne_embeddings.png"))
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
plt.savefig(SAVE_PATH, dpi=300)
print(f"📊 t-SNE plot saved to: {SAVE_PATH}")
plt.show()
