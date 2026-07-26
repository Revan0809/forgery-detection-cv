"""
train.py — Fine-tunes a pretrained ResNet18 (RGB + ELA, 4-channel input) as
a binary authentic-vs-tampered classifier.

Only conv1 (resized for the extra ELA channel), layer4, and the new
classification head are trained; everything else stays frozen. With ~350
training images and a frozen backbone this trains in well under a minute
per epoch on a laptop CPU.
"""
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import ForgeryDataset, build_model

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
LOG_PATH = os.path.join(ROOT, "training_log.csv")

EPOCHS = 8
BATCH_SIZE = 16
LR = 1e-3
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            if is_train:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def main():
    set_seed()
    torch.set_num_threads(os.cpu_count() or 4)

    train_ds = ForgeryDataset(os.path.join(DATA_DIR, "train"), augment=True)
    val_ds = ForgeryDataset(os.path.join(DATA_DIR, "val"), augment=False)
    print(f"Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model(num_extra_channels=1, freeze_until="layer4")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,}")

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LR)

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_val_acc = 0.0
    rows = []
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)
        dt = time.time() - t0
        print(f"Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  ({dt:.1f}s)")
        rows.append([epoch, train_loss, train_acc, val_loss, val_acc, round(dt, 2)])

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best_model.pt"))
            print(f"  -> new best checkpoint saved (val_acc={val_acc:.4f})")

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_seconds"])
        writer.writerows(rows)

    total_time = time.time() - start
    print(f"\nTraining complete in {total_time:.1f}s. Best val acc: {best_val_acc:.4f}")
    print(f"Checkpoint: {os.path.join(CKPT_DIR, 'best_model.pt')}")
    print(f"Training log: {LOG_PATH}")


if __name__ == "__main__":
    main()
