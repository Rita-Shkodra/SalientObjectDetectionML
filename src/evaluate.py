import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

from data_loader import get_dataloaders
from sod_model import SODModel

def iou_score(pred, target):
    pred_bin = (pred > 0.5).float()
    target = target.float()

    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection + 1e-6
    return (intersection / union).item()


def compute_metrics(pred, target):
    pred_bin = (pred > 0.5).float()

    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()

    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    iou       = iou_score(pred, target)

    return precision.item(), recall.item(), f1.item(), iou

def visualize_sample(img, mask, pred):
    img = img.permute(1, 2, 0).cpu().numpy()
    mask = mask.squeeze().cpu().numpy()
    pred = pred.squeeze().cpu().numpy()

    overlay = img.copy()
    overlay[:, :, 1] = np.maximum(overlay[:, :, 1], pred)  # add green tint
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(10,4))
    plt.subplot(1,4,1); plt.imshow(img);   plt.title("Input"); plt.axis("off")
    plt.subplot(1,4,2); plt.imshow(mask);  plt.title("Ground Truth"); plt.axis("off")
    plt.subplot(1,4,3); plt.imshow(pred);  plt.title("Prediction"); plt.axis("off")
    plt.subplot(1,4,4); plt.imshow(overlay); plt.title("Overlay"); plt.axis("off")
    plt.show()

def evaluate_model(model, test_loader, device):
    model.eval()

    total_metrics = [0,0,0,0]  
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)

            precision, recall, f1, iou = compute_metrics(preds, masks)
            total_metrics[0] += precision
            total_metrics[1] += recall
            total_metrics[2] += f1
            total_metrics[3] += iou

    batches = len(test_loader)
    avg = [m / batches for m in total_metrics]

    return {
        "Precision": avg[0],
        "Recall": avg[1],
        "F1": avg[2],
        "IoU": avg[3]
    }

