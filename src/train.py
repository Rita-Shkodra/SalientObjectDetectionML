import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data_loader import get_dataloaders
from src.sod_model import SODModel
import numpy as np
def iou(pred_mask, true_mask, threshold=0.5):
    pred_mask = (pred_mask > threshold).float()
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - intersection
    if union == 0:
        return torch.tensor(1.0)
    else:
        return intersection / union