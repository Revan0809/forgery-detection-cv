"""
data_gen.py — Synthetic forgery-detection dataset generator.

Generates ~500 image "patches" split into two classes:

  authentic  — untouched procedurally generated images (gradients, textures,
               shape scenes, document-like layouts), saved once as JPEG.
  tampered   — the same kind of source images after either:
                 * copy-move forgery: a region is copied and re-pasted
                   elsewhere in the SAME image, or
                 * splicing: a region from a DIFFERENT source image is pasted
                   in, after being independently JPEG-compressed first (so it
                   carries a different compression history than its host —
                   exactly the kind of inconsistency Error Level Analysis is
                   designed to expose).

No external images are downloaded — everything is generated with PIL/NumPy.
Images are written straight to data/{train,val,test}/{authentic,tampered}/
in a torchvision ImageFolder-compatible layout.
"""
import io
import os
import random
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

random.seed(42)
np.random.seed(42)

CANVAS = 224
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "data")
N_AUTHENTIC = 250
N_TAMPERED = 250
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15


# --------------------------------------------------------------------------
# Procedural "source" image generators
# --------------------------------------------------------------------------

def gen_gradient(size=CANVAS):
    c1 = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
    c2 = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
    horizontal = random.random() < 0.5
    ramp = np.linspace(0, 1, size, dtype=np.float32)
    arr = np.zeros((size, size, 3), dtype=np.float32)
    for ch in range(3):
        line = c1[ch] + (c2[ch] - c1[ch]) * ramp
        arr[:, :, ch] = line[np.newaxis, :] if horizontal else line[:, np.newaxis]
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def gen_noise_texture(size=CANVAS):
    small = np.random.randint(0, 255, (size // 8, size // 8, 3), dtype=np.uint8)
    img = Image.fromarray(small, "RGB").resize((size, size), Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.5)))
    return img


def gen_shapes_scene(size=CANVAS):
    bg = tuple(random.randint(180, 255) for _ in range(3))
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(5, 12)):
        shape = random.choice(["rect", "ellipse", "line"])
        color = tuple(random.randint(0, 255) for _ in range(3))
        x0, y0 = random.randint(0, size - 20), random.randint(0, size - 20)
        x1 = x0 + random.randint(10, 80)
        y1 = y0 + random.randint(10, 80)
        if shape == "rect":
            fill = color if random.random() < 0.5 else None
            draw.rectangle([x0, y0, x1, y1], outline=color, width=random.randint(1, 4), fill=fill)
        elif shape == "ellipse":
            draw.ellipse([x0, y0, x1, y1], outline=color, width=random.randint(1, 4))
        else:
            draw.line([x0, y0, x1, y1], fill=color, width=random.randint(1, 4))
    return img


def gen_document_scene(size=CANVAS):
    img = Image.new("RGB", (size, size), (250, 248, 242))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.rectangle([15, 15, size - 15, 45], outline=(20, 20, 20), width=2)
    draw.text((20, 22), "ID No. " + "".join(random.choices("0123456789", k=9)), fill=(20, 20, 20), font=font)
    for i in range(6):
        y = 60 + i * 18
        if y > size - 60:
            break
        line_len = random.randint(60, size - 40)
        draw.line([20, y, 20 + line_len, y], fill=(40, 40, 40), width=2)
    # "photo" and "signature" boxes, common on ID-style documents
    draw.rectangle([size - 70, size - 70, size - 20, size - 20], outline=(0, 0, 0), width=2)
    draw.ellipse([size - 65, size - 65, size - 25, size - 25], outline=(90, 90, 90))
    draw.line([20, size - 30, 120, size - 30], fill=(30, 30, 30), width=1)
    draw.text((20, size - 20), "SIGNATURE", fill=(60, 60, 60), font=font)
    return img


GENERATORS = [gen_gradient, gen_noise_texture, gen_shapes_scene, gen_document_scene]


def make_source_pool(n):
    return [GENERATORS[i % len(GENERATORS)]() for i in range(n)]


# --------------------------------------------------------------------------
# Tampering operations
# --------------------------------------------------------------------------

def random_box(canvas, min_s=28, max_s=64):
    s = random.randint(min_s, max_s)
    x0 = random.randint(0, canvas - s)
    y0 = random.randint(0, canvas - s)
    return x0, y0, s


def feathered_paste(base, patch, x, y):
    s = patch.size[0]
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).ellipse([2, 2, s - 2, s - 2], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
    base.paste(patch, (x, y), mask)


def add_noise(img, sigma):
    arr = np.array(img).astype(np.float32)
    arr += np.random.normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def copy_move_forgery(src):
    img = src.copy()
    x0, y0, s = random_box(CANVAS)
    patch = img.crop((x0, y0, x0 + s, y0 + s))
    if random.random() < 0.5:
        patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
    x1, y1 = x0, y0
    for _ in range(20):
        cx, cy = random.randint(0, CANVAS - s), random.randint(0, CANVAS - s)
        if abs(cx - x0) > s // 2 or abs(cy - y0) > s // 2:
            x1, y1 = cx, cy
            break
    feathered_paste(img, patch, x1, y1)
    return add_noise(img, sigma=random.uniform(1.5, 4.0))


def splicing_forgery(host, donor):
    img = host.copy()
    dx0, dy0, s = random_box(CANVAS)
    patch = donor.crop((dx0, dy0, dx0 + s, dy0 + s))
    # give the donor patch its own, independent compression history before
    # pasting -- this is the double-compression cue ELA is designed to catch
    buf = io.BytesIO()
    patch.save(buf, format="JPEG", quality=random.randint(60, 85))
    buf.seek(0)
    patch = Image.open(buf).convert("RGB")
    x1, y1 = random.randint(0, CANVAS - s), random.randint(0, CANVAS - s)
    feathered_paste(img, patch, x1, y1)
    return add_noise(img, sigma=random.uniform(1.0, 3.0))


# --------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------

def split_and_save(imgs, label):
    n = len(imgs)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    splits = {
        "train": imgs[:n_train],
        "val": imgs[n_train:n_train + n_val],
        "test": imgs[n_train + n_val:],
    }
    for split, subset in splits.items():
        for idx, im in enumerate(subset):
            path = os.path.join(OUT_DIR, split, label, f"{label}_{idx:04d}.jpg")
            im.save(path, format="JPEG", quality=random.randint(85, 95))


def build_dataset():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    for split in ("train", "val", "test"):
        for cls in ("authentic", "tampered"):
            os.makedirs(os.path.join(OUT_DIR, split, cls), exist_ok=True)

    authentic_imgs = []
    for i in range(N_AUTHENTIC):
        img = GENERATORS[i % len(GENERATORS)]()
        authentic_imgs.append(add_noise(img, sigma=random.uniform(0.5, 2.0)))

    source_pool = make_source_pool(24)
    n_cm = N_TAMPERED // 2
    n_sp = N_TAMPERED - n_cm
    tampered_imgs = [copy_move_forgery(random.choice(source_pool)) for _ in range(n_cm)]
    for _ in range(n_sp):
        host, donor = random.sample(source_pool, 2)
        tampered_imgs.append(splicing_forgery(host, donor))

    random.shuffle(authentic_imgs)
    random.shuffle(tampered_imgs)

    split_and_save(authentic_imgs, "authentic")
    split_and_save(tampered_imgs, "tampered")

    print(f"Dataset written to {OUT_DIR}")
    for split in ("train", "val", "test"):
        for cls in ("authentic", "tampered"):
            d = os.path.join(OUT_DIR, split, cls)
            print(f"  {split}/{cls}: {len(os.listdir(d))} images")


if __name__ == "__main__":
    build_dataset()
