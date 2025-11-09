import os
import time
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# =========================
# Command-line arguments
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../data")))
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--arch", choices=["mobilenetv3_small", "efficientnet_b0"], default="mobilenetv3_small")
parser.add_argument("--out", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../models")))
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =========================
# Data transforms
# =========================
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=train_tf)
test_ds = datasets.ImageFolder(os.path.join(args.data_dir, "test"), transform=test_tf)

train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

num_classes = len(train_ds.classes)
print(f"✅ Loaded dataset with {num_classes} classes")

# =========================
# Build model
# =========================
if args.arch == "mobilenetv3_small":
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_f = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_f, num_classes)
elif args.arch == "efficientnet_b0":
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, num_classes)

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# =========================
# Training loop
# =========================
for epoch in range(1, args.epochs + 1):
    model.train()
    running_loss, correct, total = 0, 0, 0
    start = time.time()

    for imgs, labels in train_dl:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    # Evaluate
    model.eval()
    correct_test, total_test = 0, 0
    with torch.no_grad():
        for imgs, labels in test_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct_test += (preds == labels).sum().item()
            total_test += labels.size(0)
    test_acc = 100 * correct_test / total_test

    print(f"Epoch [{epoch}/{args.epochs}] "
          f"Loss: {running_loss/len(train_dl):.4f} "
          f"Train Acc: {train_acc:.2f}% "
          f"Test Acc: {test_acc:.2f}% "
          f"({time.time()-start:.1f}s)")

# =========================
# Save model
# =========================
os.makedirs(args.out, exist_ok=True)
out_path = os.path.join(args.out, f"agrisense_{args.arch}.pth")
torch.save(model.state_dict(), out_path)
print(f"💾 Model saved to: {out_path}")
