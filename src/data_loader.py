import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SODDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))
        self.masks  = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = img.astype(np.float32) / 255.0 

        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (128, 128))
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)

        img = torch.from_numpy(img).permute(2, 0, 1)  
        mask = torch.from_numpy(mask)

        return img, mask

def get_dataloaders(train_dir, val_dir, batch_size=16):
    train_dataset = SODDataset(os.path.join(train_dir, "images"),
                               os.path.join(train_dir, "masks"))
    val_dataset   = SODDataset(os.path.join(val_dir, "images"),
                               os.path.join(val_dir, "masks"))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
