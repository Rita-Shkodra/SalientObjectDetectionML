import os
import shutil
import random

RAW_DIR = "/content/drive/MyDrive/SOD_Data/raw"
PROCESSED_DIR = "/content/drive/MyDrive/SOD_Data/processed"

TR_IMG_DIR = os.path.join(RAW_DIR, "DUTS-TR", "DUTS-TR", "DUTS-TR-Image")
TR_MASK_DIR = os.path.join(RAW_DIR, "DUTS-TR", "DUTS-TR", "DUTS-TR-Mask")

TE_IMG_DIR = os.path.join(RAW_DIR, "DUTS-TE", "DUTS-TE", "DUTS-TE-Image")
TE_MASK_DIR = os.path.join(RAW_DIR, "DUTS-TE", "DUTS-TE", "DUTS-TE-Mask")

for split in ["train", "val", "test"]:
    for kind in ["images", "masks"]:
        os.makedirs(os.path.join(PROCESSED_DIR, split, kind), exist_ok=True)


def collect_pairs(img_dir, mask_dir):
    imgs = sorted(os.listdir(img_dir))
    masks = sorted(os.listdir(mask_dir))
    out = []
    for img in imgs:
        mask = img.replace(".jpg", ".png")
        if mask in masks:
            out.append((img_dir, img, mask_dir, mask))
    return out


pairs_tr = collect_pairs(TR_IMG_DIR, TR_MASK_DIR)
pairs_te = collect_pairs(TE_IMG_DIR, TE_MASK_DIR)

pairs = pairs_tr + pairs_te
print("Total paired samples found:", len(pairs))

random.shuffle(pairs)

total = len(pairs)
train_len = int(0.7 * total)
val_len = int(0.15 * total)

train_pairs = pairs[:train_len]
val_pairs = pairs[train_len : train_len + val_len]
test_pairs = pairs[train_len + val_len :]

print("Split sizes:")
print(" Train:", len(train_pairs))
print(" Val:", len(val_pairs))
print(" Test:", len(test_pairs))


def copy_pairs(pairs, split):
    img_dest = os.path.join(PROCESSED_DIR, split, "images")
    mask_dest = os.path.join(PROCESSED_DIR, split, "masks")

    for img_dir, img_name, mask_dir, mask_name in pairs:
        shutil.copy(
            os.path.join(img_dir, img_name),
            os.path.join(img_dest, img_name)
        )
        shutil.copy(
            os.path.join(mask_dir, mask_name),
            os.path.join(mask_dest, mask_name)
        )


copy_pairs(train_pairs, "train")
copy_pairs(val_pairs, "val")
copy_pairs(test_pairs, "test")

print("Dataset preprocessing completed")
