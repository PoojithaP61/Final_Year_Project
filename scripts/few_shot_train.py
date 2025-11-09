import os
import time
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

# ======================
# Configuration
# ======================
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/train"))
SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/few_shot_encoder.pth"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Data transforms
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ======================
# Dataset
# ======================
train_ds = datasets.ImageFolder(DATA_DIR, transform=transform)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
num_classes = len(train_ds.classes)
print(f"✅ Loaded dataset with {num_classes} classes.")

# ======================
# Feature extractor (MobileNetV3)
# ======================
backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
feature_dim = 576  # MobileNetV3-Small output dimension
backbone.classifier = nn.Identity()  # remove classification head
backbone = backbone.to(DEVICE)

# ======================
# Projection Head (Contrastive)
# ======================
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=576, out_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        # Flatten feature maps if needed
        if x.ndim > 2:
            x = torch.flatten(x, 1)
        return self.fc(x)

proj_head = ProjectionHead(in_dim=feature_dim, out_dim=128).to(DEVICE)

# ======================
# Supervised Contrastive Loss
# ======================
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        sim = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - logits_max.detach()
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos.mean()
        return loss

criterion = SupConLoss()
optimizer = torch.optim.Adam(
    list(backbone.parameters()) + list(proj_head.parameters()), lr=1e-4
)

# ======================
# Training loop
# ======================
print(f"Training contrastive encoder on {num_classes} classes ...")
for epoch in range(10):
    backbone.train()
    proj_head.train()
    running_loss = 0
    start = time.time()

    for imgs, labels in train_dl:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        feats = backbone(imgs)
        feats = nn.functional.normalize(proj_head(feats), dim=1)
        loss = criterion(feats, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/10] Loss: {running_loss/len(train_dl):.4f} ({time.time()-start:.1f}s)")

# ======================
# Save model
# ======================
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
torch.save(backbone.state_dict(), SAVE_PATH)
print(f"💾 Encoder saved successfully at: {SAVE_PATH}")
