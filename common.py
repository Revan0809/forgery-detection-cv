"""
common.py — shared ELA preprocessing, dataset, and model-building code used
by both train.py and eval.py.
"""
import io
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageChops
from torch.utils.data import Dataset
from torchvision import models, transforms

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def compute_ela(pil_img, quality=90, scale=15):
    """Error Level Analysis: re-save the image at a known JPEG quality and
    return the amplified per-pixel difference (as a single-channel image).

    Regions that were edited after the image's last save -- or copied in
    from a source with a different compression history -- re-compress
    differently than untouched regions, showing up as a distinct error
    level. This is a standard forensic cue for detecting splicing/copy-move
    tampering in JPEG images.
    """
    pil_img = pil_img.convert("RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")
    diff = ImageChops.difference(pil_img, recompressed)
    arr = np.array(diff).astype(np.float32) * scale
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB").convert("L")


class ForgeryDataset(Dataset):
    """Loads (RGB image, ELA map) pairs stacked into a 4-channel tensor."""

    CLASSES = ["authentic", "tampered"]

    def __init__(self, root, ela_quality=90, ela_scale=15, augment=False):
        self.samples = []
        for label, cls in enumerate(self.CLASSES):
            cls_dir = os.path.join(root, cls)
            for fn in sorted(os.listdir(cls_dir)):
                self.samples.append((os.path.join(cls_dir, fn), label))
        self.ela_quality = ela_quality
        self.ela_scale = ela_scale
        self.augment = augment
        self.img_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.ela_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        ela = compute_ela(img, quality=self.ela_quality, scale=self.ela_scale)
        if self.augment and np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            ela = ela.transpose(Image.FLIP_LEFT_RIGHT)
        img_t = self.img_tf(img)
        ela_t = self.ela_tf(ela)
        x = torch.cat([img_t, ela_t], dim=0)  # 4 x H x W (RGB + ELA)
        return x, label


def build_model(num_extra_channels=1, freeze_until="layer4"):
    """ResNet18 pretrained on ImageNet, adapted to take a 4-channel
    (RGB + ELA) input, with everything before `freeze_until` frozen.

    The pretrained conv1 weights are kept for the RGB channels; the new ELA
    channel's weights are initialized as the mean of the RGB filters, a
    reasonable starting point instead of random noise.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    old_conv = model.conv1
    new_conv = nn.Conv2d(
        3 + num_extra_channels, old_conv.out_channels,
        kernel_size=old_conv.kernel_size, stride=old_conv.stride,
        padding=old_conv.padding, bias=False,
    )
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        extra = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight[:, 3:] = extra.repeat(1, num_extra_channels, 1, 1)
    model.conv1 = new_conv

    for param in model.parameters():
        param.requires_grad = False
    unfreeze = False
    for name, module in model.named_children():
        if name == freeze_until:
            unfreeze = True
        if unfreeze:
            for p in module.parameters():
                p.requires_grad = True
    for p in model.conv1.parameters():  # conv1 has fresh, untrained weights
        p.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, 2)
    for p in model.fc.parameters():
        p.requires_grad = True

    return model
