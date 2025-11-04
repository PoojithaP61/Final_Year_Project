import os, shutil, random
from tqdm import tqdm

SOURCE_DIR = r"../data/PlantVillage"
TRAIN_DIR = r"../data/train"
TEST_DIR = r"../data/test"

split_ratio = 0.8

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

for category in tqdm(os.listdir(SOURCE_DIR)):
    cat_path = os.path.join(SOURCE_DIR, category)
    if not os.path.isdir(cat_path):
        continue

    images = os.listdir(cat_path)
    random.shuffle(images)
    split_point = int(len(images) * split_ratio)
    train_imgs = images[:split_point]
    test_imgs = images[split_point:]

    os.makedirs(os.path.join(TRAIN_DIR, category), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, category), exist_ok=True)

    for img in train_imgs:
        shutil.copy(os.path.join(cat_path, img), os.path.join(TRAIN_DIR, category, img))
    for img in test_imgs:
        shutil.copy(os.path.join(cat_path, img), os.path.join(TEST_DIR, category, img))
