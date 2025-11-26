import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from data_loader import get_dataloaders
from sod_model import SODModel


bce_loss = nn.BCELoss()


def iou_score(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection + 1e-6
    return intersection / union


def combined_loss(pred, target):
    bce = bce_loss(pred, target)
    iou = iou_score(pred, target)
    return bce + 0.5 * (1.0 - iou)


def compute_metrics(pred, target):
    pred_bin = (pred > 0.5).float()

    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = iou_score(pred_bin, target)

    return precision.item(), recall.item(), f1.item(), iou.item()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

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
    total_loss = 0.0
    total_metrics = [0.0, 0.0, 0.0, 0.0]

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


def save_checkpoint(epoch, model, optimizer, scheduler, path):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict()
    }, path)
    print(f"Checkpoint saved at {path}")


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = ckpt["epoch"] + 1
    print(f"Resume training from epoch {start_epoch}")
    return start_epoch


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    train_loader, val_loader, _ = get_dataloaders(
        "/content/SOD_Data/processed/train",
        "/content/SOD_Data/processed/val",
        "/content/SOD_Data/processed/test",
        batch_size=16,
    )

    model = SODModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    best_val_loss = float("inf")
    checkpoint_path = "checkpoint.pth"
    best_model_path = "best_model.pth"

    start_epoch = 1
    EPOCHS = 25

    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)

    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, metrics = val_epoch(model, val_loader, device)

        precision, recall, f1, iou = metrics

        print(f"\nEpoch {epoch} Summary:")
        print(f" Train Loss: {train_loss:.4f}")
        print(f" Val Loss:   {val_loss:.4f}")
        print(f" Precision:  {precision:.4f}")
        print(f" Recall:     {recall:.4f}")
        print(f" F1-score:   {f1:.4f}")
        print(f" IoU:        {iou:.4f}")

        scheduler.step(val_loss)

        save_checkpoint(epoch, model, optimizer, scheduler, checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at {best_model_path}")

    print("\nTraining complete")


if __name__ == "__main__":
    main()
