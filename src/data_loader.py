import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import random

class SODDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def safe_load_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            return mask
        try:
            pil_img = Image.open(path).convert("L")
            return np.array(pil_img)
        except:
            return None

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = self.masks[idx]

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = self.safe_load_mask(mask_path)
        if mask is None:
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)

        img = cv2.resize(img, (128, 128))
        mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        mask = np.expand_dims(mask, axis=0)

        if self.augment:
            if random.random() > 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            factor = random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 1)

        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask)

        return img, mask


def get_dataloaders(train_dir, val_dir, test_dir, batch_size=16):
    train_dataset = SODDataset(
        os.path.join(train_dir, "images"),
        os.path.join(train_dir, "masks"),
        augment=True
    )

    val_dataset = SODDataset(
        os.path.join(val_dir, "images"),
        os.path.join(val_dir, "masks"),
        augment=False
    )

    test_dataset = SODDataset(
        os.path.join(test_dir, "images"),
        os.path.join(test_dir, "masks"),
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
