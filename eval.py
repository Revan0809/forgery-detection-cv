"""
eval.py — Evaluates the best checkpoint on the held-out test set: accuracy,
precision, recall, F1, a confusion matrix, and a grid of example predictions.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              precision_recall_fscore_support)
from torch.utils.data import DataLoader

from common import IMAGENET_MEAN, IMAGENET_STD, ForgeryDataset, build_model

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
CKPT_PATH = os.path.join(ROOT, "checkpoints", "best_model.pt")
OUT_DIR = os.path.join(ROOT, "outputs")


def denormalize(img_t):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (img_t * std + mean).clamp(0, 1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    test_ds = ForgeryDataset(os.path.join(DATA_DIR, "test"), augment=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_ds)}")

    model = build_model(num_extra_channels=1, freeze_until="layer4")
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    summary = (
        f"Test set results\n"
        f"  Accuracy:  {acc:.4f}\n"
        f"  Precision: {precision:.4f}\n"
        f"  Recall:    {recall:.4f}\n"
        f"  F1 score:  {f1:.4f}\n"
        f"  Confusion matrix (rows=true, cols=pred) [authentic, tampered]:\n{cm}\n"
    )
    print(summary)
    with open(os.path.join(OUT_DIR, "eval_metrics.txt"), "w") as f:
        f.write(summary)

    # --- confusion matrix figure ---
    classes = ForgeryDataset.CLASSES
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(classes)
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"))
    plt.close(fig)

    # --- example predictions grid ---
    n_examples = min(12, len(test_ds))
    idxs = np.random.RandomState(0).choice(len(test_ds), size=n_examples, replace=False)
    cols = 4
    rows_n = int(np.ceil(n_examples / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3, rows_n * 3))
    axes = np.array(axes).reshape(-1)
    with torch.no_grad():
        for ax_i, idx in enumerate(idxs):
            x, y = test_ds[idx]
            out = model(x.unsqueeze(0))
            prob = torch.softmax(out, dim=1)[0]
            pred = int(prob.argmax())
            img_disp = denormalize(x[:3]).permute(1, 2, 0).numpy()
            ax = axes[ax_i]
            ax.imshow(img_disp)
            color = "green" if pred == y else "red"
            ax.set_title(f"T:{classes[y]}\nP:{classes[pred]} ({prob[pred]:.2f})",
                         color=color, fontsize=9)
            ax.axis("off")
    for ax_i in range(len(idxs), len(axes)):
        axes[ax_i].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "example_predictions.png"))
    plt.close(fig)

    print(f"Saved: {os.path.join(OUT_DIR, 'eval_metrics.txt')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'confusion_matrix.png')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'example_predictions.png')}")


if __name__ == "__main__":
    main()
