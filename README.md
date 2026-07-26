# forgery-detection-cv

A small, fully-synthetic, CPU-only pipeline that fine-tunes a pretrained
ResNet18 to classify image patches as **authentic** or **tampered**
(copy-move / splicing forgery). The task is directly relevant to
**document and identity fraud detection**: passports, ID cards, and other
scanned documents are frequently attacked by copying a region within the
same image (e.g. duplicating a stamp) or splicing in content from another
document/photo (e.g. swapping a photo or altering a printed field).

Everything runs top-to-bottom on a laptop CPU in a few minutes, with no
external dataset downloads — all images are generated procedurally with
PIL/NumPy.

## Problem

Manipulated identity documents and photos are a core vector for fraud in
KYC (know-your-customer), onboarding, and verification pipelines. Two of
the most common low-effort tampering techniques are:

- **Copy-move forgery** — a region of an image is copied and pasted
  elsewhere in the *same* image (e.g. cloning a background over a mismatched
  photo, duplicating a security pattern to hide an edit).
- **Splicing** — a region from a *different* image is pasted in (e.g.
  swapping in another person's photo, altering a printed name/number field
  using text lifted from elsewhere).

Both operations tend to leave forensic traces even when visually
convincing: edge blending artifacts, and — critically — **inconsistent JPEG
compression history** between the pasted region and its surroundings.

## Approach

### Transfer learning

We start from a ResNet18 pretrained on ImageNet (`torchvision`) rather than
training from scratch. With only ~500 synthetic examples, a randomly
initialized CNN would badly overfit; a pretrained backbone already knows
general-purpose edge/texture/color features that transfer well to spotting
visual seams and blending artifacts.

We **freeze everything up through `layer3`** and only fine-tune `layer4`
plus a new binary classification head (`fc`). Rationale:

- Early/mid layers of a pretrained CNN encode generic, transferable
  features (edges, textures, simple shapes) that are already well-suited to
  this task and don't need to change.
- With a small dataset, updating all ~11M parameters risks overfitting and
  is unnecessary; updating only the last block (~8.4M of the trainable
  budget, dominated by `layer4`) and the head lets the model specialize to
  forgery-specific cues while keeping training fast and stable on CPU.
- `conv1` is also fine-tuned, but only because its shape had to change (see
  below) — its extra channel starts from an untrained initialization and
  needs to learn.

### Error Level Analysis (ELA) as an extra input channel

[Error Level Analysis](https://en.wikipedia.org/wiki/Error_level_analysis)
is a classical forensic technique: re-save a JPEG at a known quality level
and diff it against the original. Untouched regions — which share one
consistent compression history — settle into a stable, low, uniform error
level. Regions that were edited after the image's last save, or spliced in
from a source with a *different* compression history, tend to re-compress
differently and show a distinct error level.

`common.compute_ela()` implements this: recompress at quality 90, take the
per-pixel absolute difference, amplify it, and reduce to a single grayscale
channel. This ELA map is concatenated onto the RGB image, producing a
**4-channel (R, G, B, ELA) input**. The pretrained `conv1` (which expects
3 channels) is expanded to 4 channels; the 3 RGB filters keep their
pretrained weights, and the new ELA filter is initialized as the mean of
the RGB filters (a much better starting point than random noise) and then
learned during fine-tuning.

To make sure ELA actually carries signal in the synthetic data, the
`splicing_forgery` generator independently JPEG-compresses the donor patch
*before* pasting it into the host image — a genuine double-compression
mismatch, not a cosmetic effect.

### Synthetic dataset

Since no external dataset may be downloaded, `data_gen.py` procedurally
builds ~500 224x224 image patches from scratch using PIL/NumPy:

- **Authentic** (250): gradients, blurred noise textures, random shape
  scenes, and ID-document-like layouts (text lines, a "photo" box, a
  "signature" box) drawn with `ImageDraw`/`ImageFont`.
- **Tampered** (250, split ~evenly):
  - *Copy-move*: a random 28–64px region is copied, optionally flipped, and
    feather-pasted elsewhere in the same source image, plus mild Gaussian
    noise.
  - *Splicing*: a random region from a *different* source image is
    JPEG-compressed on its own (simulating an independent capture/edit
    history), then feather-pasted into the host image, plus mild noise.

Images are saved as JPEGs (quality 85–95) and split 70/15/15 into
`data/{train,val,test}/{authentic,tampered}/` — a standard
`ImageFolder`-style layout.

## Results

*(from an actual run of `python data_gen.py && python train.py && python eval.py` on a laptop CPU, no GPU)*

- Dataset: 350 train / 74 val / 76 test images (balanced classes)
- Training: 8 epochs, batch size 16, Adam (lr=1e-3), ~270s total (~34s/epoch) on CPU
- Best val accuracy: **89.19%** (epoch 7)

**Test set metrics** (`outputs/eval_metrics.txt`):

| Metric | Value |
|---|---|
| Accuracy | 0.8289 |
| Precision (tampered) | 0.9310 |
| Recall (tampered) | 0.7105 |
| F1 score (tampered) | 0.8060 |

Confusion matrix (rows = true, cols = predicted; order = [authentic, tampered]):

```
[[36  2]
 [11 27]]
```

Precision is notably higher than recall: the model is conservative about
flagging tampering (few false alarms on authentic images: 2/38), but misses
some subtler forgeries (11/38 tampered images predicted authentic) —
sensible given the small, synthetic training set and short training run.
See `outputs/confusion_matrix.png` and `outputs/example_predictions.png`
for visuals, and `training_log.csv` for the full per-epoch curve.

## Reproduce

```bash
pip install -r requirements.txt
python data_gen.py && python train.py && python eval.py
```

Total runtime end-to-end: a few minutes on a laptop CPU (data generation:
seconds; training: ~4–5 minutes; evaluation: seconds).

## Project layout

```
forgery-detection-cv/
  data_gen.py          # synthetic dataset generator (authentic + tampered)
  common.py            # ELA preprocessing, Dataset, model construction
  train.py             # fine-tuning loop, checkpointing, CSV logging
  eval.py              # test-set metrics, confusion matrix, prediction grid
  requirements.txt
  data/                # generated by data_gen.py (train/val/test splits)
  checkpoints/          # best_model.pt, written by train.py
  outputs/              # eval_metrics.txt, confusion_matrix.png, example_predictions.png
  training_log.csv      # per-epoch train/val loss & accuracy
```

## Limitations

This is a small, self-contained demo built to run without internet access
or a GPU — not a production forgery detector. The synthetic tampering
(feathered pastes on procedurally generated scenes) is much cleaner than
real-world forgeries, and the dataset is small (500 images from a limited
set of scene templates). For a production identity-document fraud pipeline
you'd want: real (or much more diverse/adversarial) tampered examples,
larger-scale training, additional forensic cues (noise-level analysis,
JPEG block-grid analysis, PRNU), and careful evaluation for demographic
and document-type fairness.
