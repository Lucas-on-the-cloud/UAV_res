# Plan A — Density-aware Training Crops + SAHI Inference

## Motivation

Advisor feedback from 2026-05-13:

> "Tăng accuracy bằng clustering và cropping image (chia 2 phần) dựa trên density"

The HERIDAL baseline (YOLOv12-s, mAP@0.5 = 0.759, mAP@0.5:0.95 = 0.344) showed acceptable detection but **poor localization on tiny persons**. This is a known issue when training on resized aerial imagery: persons at ~30 px in the original 4000×3000 image become ~5 px after resizing to 640. The model never learns to localize larger person patches.

A naive remedy is **SAHI sliced inference at native resolution**. We attempted this first and it failed catastrophically (mAP@0.5: 0.759 → 0.255). Diagnosis: training–inference distribution shift. The model was never shown native-resolution patches, so when SAHI passed them in, the model hallucinated 354 detections on an image with 1 person.

Plan A fixes this by **moving the density-aware step into training**: generate native-resolution crops from the original images and train the detector on those crops.

## Method

### 1. Crop generation

For each training image (4000×3000):
- **2 person-centered crops per ground-truth person** with random offset of up to ±160 px (jitter so the person isn't always exactly centered).
- **1 random crop** per image (may or may not contain persons — helps the model learn to suppress background false positives).
- A bounding box is kept only if more than 50% of its area falls inside the crop.
- All crops are 640×640 (matches the YOLOv12-s training input).

Output: ~4,500 training crops + ~600 validation crops, all at native resolution (no resize).

### 2. Training

YOLOv12-s initialized from COCO-pretrained weights, fine-tuned on the crops dataset for 80 epochs (AdamW, lr=0.001, batch=16, AMP, Tesla T4 commit-and-run).

### 3. Inference

SAHI sliced inference on the **original 4000×3000 validation images** (not the crops):
- Slice 640×640 with 0.2 overlap (about 48 tiles per image)
- `perform_standard_pred=False` (sliced only)
- `postprocess_type="NMS"` with match threshold 0.5

Evaluation: pycocotools COCO eval on original images (apples-to-apples with SAHI predictions in original-image coordinates).

## Results

| Metric | Baseline (resized train) | Naive SAHI (resized train) | **Plan A (crops train + SAHI)** | Δ vs Baseline |
|---|---|---|---|---|
| mAP@0.5 | 0.7586 | 0.2549 | **0.8721** | **+0.1135** |
| mAP@0.5:0.95 | 0.3444 | 0.1285 | **0.5735** | **+0.2291** |
| mAP@0.75 | — | 0.1169 | **0.6611** | (large) |
| AP_small | — | 0.0016 | **0.494** | (~250× over naive SAHI) |
| AR_100 | — | 0.4384 | **0.6616** | +0.224 |

### Key takeaways

1. **mAP@0.5 improves by +0.11** absolute (+15% relative) — model finds more persons.
2. **mAP@0.5:0.95 improves by +0.23** absolute (+66.5% relative) — model localizes them much more precisely.
3. **AP_small jumps from essentially zero to 0.494** — the tiny-person problem is substantially resolved.
4. **mAP@0.75 = 0.661** — even at strict IoU thresholds the model produces accurate bounding boxes, which matters for converting detections to real-world GPS coordinates in SAR.

## Why training-time density crops succeed where inference-time slicing alone fails

| Aspect | Baseline | Naive SAHI | Plan A |
|---|---|---|---|
| Training input | resized 4000×3000 → 640 | resized 4000×3000 → 640 | native-resolution 640×640 crops |
| Inference input | resized 4000×3000 → 640 | 640×640 SAHI tiles (native res) | 640×640 SAHI tiles (native res) |
| Train ↔ inference resolution | match (both small persons) | **mismatch** | **match** (both native persons) |
| Outcome | OK | catastrophic hallucination | best of both worlds |

The lesson: density-aware *inference* is only useful when the model has also seen density-aware *training* samples. The two have to be done together.

## Implications for the project

- **HERIDAL working baseline is now Plan A** (mAP@0.5 = 0.872), not the original full-image-trained model.
- The mAP@0.5:0.95 issue identified earlier is largely resolved.
- The next contribution to add is **AirSim synthetic-data augmentation** focusing on hard SAR scenarios (occlusion, lying poses, dense vegetation, low-light). With the density-crop pipeline already in place, synthetic crops can be mixed in directly.

## Files

- Final weights: `yolov12s_heridal_crops_BEST.pt` (local + Google Drive)
- Metrics: [`plan_a_density_crops_result.json`](plan_a_density_crops_result.json)
- Training notebook: see Kaggle commit `plan-a-density-crops-train` (private)

## Reproducibility

To reproduce Plan A:
1. Download HERIDAL via Roboflow API (`hung1244s-workspace/heridal-lrbkc-8vnfq`, v1)
2. Generate 640×640 crops with the strategy above (script in commit history)
3. Train YOLOv12-s for 80 epochs with the configuration in the JSON
4. Run SAHI sliced inference on the original 4000×3000 val images
5. Evaluate with pycocotools

Total compute: ~3 hours on a Tesla T4 (Kaggle free tier).
