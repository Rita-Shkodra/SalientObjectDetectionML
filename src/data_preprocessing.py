import os
import shutil
import random

RAW_DIR = "/content/drive/MyDrive/SOD_Data/raw"
PROCESSED_DIR = "/content/drive/MyDrive/SOD_Data/processed"

TR_IMG_DIR = os.path.join(RAW_DIR, "DUTS-TR", "DUTS-TR", "DUTS-TR-Image")
TR_MASK_DIR = os.path.join(RAW_DIR, "DUTS-TR", "DUTS-TR", "DUTS-TR-Mask")

for split in ["train", "val", "test"]:
    for kind in ["images", "masks"]:
        os.makedirs(os.path.join(PROCESSED_DIR, split, kind), exist_ok=True)


images = sorted(os.listdir(TR_IMG_DIR))
masks  = sorted(os.listdir(TR_MASK_DIR))

pairs = []

for img_name in images:
    mask_name = img_name.replace(".jpg", ".png")  # DUTS naming
    if mask_name in masks:
        pairs.append((img_name, mask_name))

print(f"Total paired samples found: {len(pairs)}")


random.shuffle(pairs)

total = len(pairs)
train_len = int(0.7 * total)
val_len   = int(0.15 * total)

train_pairs = pairs[:train_len]
val_pairs   = pairs[train_len : train_len + val_len]
test_pairs  = pairs[train_len + val_len :]

print("Split sizes:")
print(" Train:", len(train_pairs))
print(" Val:",   len(val_pairs))
print(" Test:",  len(test_pairs))


def move_pairs(pairs, split_name):
    img_dest = os.path.join(PROCESSED_DIR, split_name, "images")
    mask_dest = os.path.join(PROCESSED_DIR, split_name, "masks")

    for img_name, mask_name in pairs:
        shutil.copy(
            os.path.join(TR_IMG_DIR, img_name),
            os.path.join(img_dest, img_name)
        )
        shutil.copy(
            os.path.join(TR_MASK_DIR, mask_name),
            os.path.join(mask_dest, mask_name)
        )

move_pairs(train_pairs, "train")
move_pairs(val_pairs, "val")
move_pairs(test_pairs, "test")

print("Dataset preprocessing completed")
