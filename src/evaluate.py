import torch
import matplotlib.pyplot as plt
import numpy as np

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
    target = target.float()
    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = iou_score(pred, target)
    return float(precision), float(recall), float(f1), float(iou)


def visualize_sample(img, mask, pred):
    img = img.permute(1, 2, 0).cpu().numpy()
    mask = mask.squeeze().cpu().numpy()
    pred = pred.squeeze().cpu().numpy()
    overlay = img.copy()
    overlay[:, :, 1] = np.maximum(overlay[:, :, 1], pred)
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 4, 1); plt.imshow(img); plt.title("Input"); plt.axis("off")
    plt.subplot(1, 4, 2); plt.imshow(mask); plt.title("Ground Truth"); plt.axis("off")
    plt.subplot(1, 4, 3); plt.imshow(pred); plt.title("Prediction"); plt.axis("off")
    plt.subplot(1, 4, 4); plt.imshow(overlay); plt.title("Overlay"); plt.axis("off")
    plt.show()


def evaluate_model(model, test_loader, device):
    model.eval()
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_iou = 0.0
    count = 0

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)

            precision, recall, f1, iou = compute_metrics(preds, masks)

            if np.isnan(precision) or np.isnan(recall) or np.isnan(f1) or np.isnan(iou):
                continue

            total_precision += precision
            total_recall += recall
            total_f1 += f1
            total_iou += iou
            count += 1

    if count == 0:
        raise ValueError("No valid samples in test set")

    return {
        "precision": total_precision / count,
        "recall": total_recall / count,
        "f1": total_f1 / count,
        "iou": total_iou / count,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    _, _, test_loader = get_dataloaders(
        "/content/drive/MyDrive/SOD_Data/processed/train",
        "/content/drive/MyDrive/SOD_Data/processed/val",
        "/content/drive/MyDrive/SOD_Data/processed/test",
        batch_size=1
    )

    model = SODModel().to(device)
    model.load_state_dict(
    torch.load("/content/drive/MyDrive/SOD_Project/best_model.pth", map_location=device)
)

    print("Loaded best_model.pth")

    results = evaluate_model(model, test_loader, device)

    print("\nTest Set Performance:")
    print(f" Precision: {results['precision']:.4f}")
    print(f" Recall:    {results['recall']:.4f}")
    print(f" F1-score:  {results['f1']:.4f}")
    print(f" IoU:       {results['iou']:.4f}")

    count = 3
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            visualize_sample(imgs[0], masks[0], preds[0])
            count -= 1
            if count == 0:
                break


if __name__ == "__main__":
    main()
