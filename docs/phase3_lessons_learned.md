# Phase 3 — Lessons Learned: Failed Attempts at Synthetic-Data Augmentation

**Date:** 2026-05-15
**Status:** Phase 3 attempted, abandoned. Plan A (Phase 2.5) remains the working contribution.

## Summary

We attempted two synthetic-data augmentation approaches in Phase 3, both of which produced near-zero mAP on the HERIDAL validation set under SAHI sliced inference. Below is the diagnosis and the resulting takeaway that informs the eventual paper write-up.

## Setup

- Hardware: rented VM with NVIDIA RTX 5090 (32 GB VRAM), Windows 10, CUDA 12.8, PyTorch nightly.
- Baseline to beat: Plan A (Phase 2.5) — YOLOv12-s trained on 640×640 native-resolution density-aware crops from HERIDAL train, then evaluated with SAHI sliced inference on the original 4000×3000 HERIDAL val. Result: **mAP@0.5 = 0.872, mAP@0.5:0.95 = 0.574, AP_small = 0.494**.

## Approach A: SDXL diffusion synthetic + raw HERIDAL training

**Pipeline:**
1. Generate 2,000 aerial backgrounds with Stable Diffusion XL (1024×1024).
2. Composite HERIDAL person crops onto these backgrounds (with alpha-blended edges, random scale 25–80 px, random rotation).
3. Train YOLOv12 on **raw HERIDAL train images (4000×3000)** mixed with the 2,000 synthetic 1024×1024 images.
4. Evaluate with SAHI sliced inference (640×640 tiles, 0.2 overlap).

**Result (first variant `yolov12s_25sy`, after 80 epochs):**

```
mAP@0.5     = 0.0130
mAP@0.5:0.95 = 0.0067
AP_small    = 0.0431
AR_100      = 0.0388
```

→ Essentially zero — model failed to detect anything meaningfully at inference time.

## Approach B: HERIDAL-to-HERIDAL copy-paste + raw HERIDAL training

**Pipeline:**
1. Extract person crops from HERIDAL train.
2. Composite them onto OTHER HERIDAL train images (same domain, real photography on both sides).
3. Train YOLOv12-s on raw HERIDAL train + 2,000 same-domain composite samples.
4. Evaluate with SAHI sliced inference.

**Result:**

```
mAP@0.5     = 0.0000
mAP@0.5:0.95 = 0.0000
AP_small    = 0.0001
AR_100      = 0.0018
```

→ Also essentially zero. The "stay in domain" intuition didn't save the model.

## Diagnosis: training–inference scale mismatch

The failure mode is the **same training-vs-inference distribution mismatch** that motivated Plan A in the first place. We forgot to apply that lesson here:

- HERIDAL raw images are 4000×3000. The YOLO trainer with `imgsz=640` resizes them to 640×640, so a typical 30-pixel person in the original becomes ~5 pixels in the training input.
- SAHI inference, by contrast, slices the original 4000×3000 image into 640×640 tiles **at native resolution** — so persons appear at their actual ~30-pixel scale in each tile.
- The training distribution (mostly 5-pixel persons from resized HERIDAL raw) does not match the inference distribution (30-pixel persons in SAHI tiles). The model learns features at the wrong scale.

Plan A avoided this by training on **640×640 native-resolution crops** extracted from HERIDAL train — preserving the actual pixel scale of persons. Both Phase 3 attempts mixed in raw HERIDAL train images, which dominated the gradient signal at the wrong scale and broke SAHI compatibility.

The synthetic data we generated (SDXL composites at 1024 and HERIDAL composites at 4000×3000) was reasonable on its own, but mixing it with raw HERIDAL training images polluted the scale distribution.

## Takeaway

> Naïve mixing of synthetic data with raw full-resolution training images defeats SAHI sliced inference because the dominant scale in the training distribution does not match the scale presented at inference time. To benefit from synthetic augmentation under SAHI, the synthetic samples must be processed through the same density-aware crop pipeline as the real data, preserving native-resolution scale throughout.

## What we kept

- **Plan A weights and pipeline** remain the working state of the project (Phase 2.5).
- **Plan A results** (mAP@0.5 = 0.872) are unchanged and remain the headline contribution.
- The Phase 3 attempts are documented here as a negative result and a lesson; they are not represented as a positive contribution.

## Implications for the paper

The intended Phase 3 ablation table (synthetic vs no-synthetic) is **not** included. Instead the paper will:

1. Present Plan A (Phase 2.5) as the main contribution.
2. Briefly discuss the Phase 3 failure as evidence that synthetic-data augmentation in this regime is non-trivial and requires careful matching of training/inference scale.
3. Identify "crop-then-synthesize" (apply the density-aware crop pipeline to synthetic samples too) as concrete future work.

## What we will not do

- Continue debugging Phase 3 within the current GPU rental.
- Re-run experiments with corrected pipeline (would require another full training cycle).

The remaining rental time will be used for backing up artifacts and saving the project state. Plan A's contribution is sufficient for an honest workshop-paper-grade result.
