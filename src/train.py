import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from src.data_loader import get_dataloaders
from src.sod_model import SODModel
import numpy as np
def iou_loss(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection + 1e-6
    return intersection / union

def combined_loss(pred, target):
    bce = bce_loss(pred, target)
    iou = iou_loss(pred, target)
    return bce + 0.5 * (1 - iou)

def compute_metrics(pred, target):
    pred_bin = (pred > 0.5).float()

    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()

    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    iou       = iou_loss(pred_bin, target)

    return precision.item(), recall.item(), f1.item(), iou.item()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for imgs, masks in tqdm(loader, desc="Training"):
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        preds = model(imgs)

        loss = combined_loss(preds, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)



def val_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    total_metrics = [0,0,0,0]  

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Validating"):
            imgs, masks = imgs.to(device), masks.to(device)

            preds = model(imgs)
            loss = combined_loss(preds, masks)

            total_loss += loss.item()

            precision, recall, f1, iou = compute_metrics(preds, masks)
            total_metrics[0] += precision
            total_metrics[1] += recall
            total_metrics[2] += f1
            total_metrics[3] += iou

    num_batches = len(loader)
    avg_metrics = [m / num_batches for m in total_metrics]

    return total_loss / len(loader), avg_metrics
