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
