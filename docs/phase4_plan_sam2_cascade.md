# Phase 4 — Execution Plan: SAM2 Bbox Refinement + Hard-Negative Cascade Classifier

**Status:** planned, ready to execute
**Created:** 2026-05-20
**Baseline to beat:** Plan A (Phase 2.5) — mAP@0.5 = 0.872, mAP@0.5:0.95 = 0.574, AP_small = 0.494

## Goal

Stack two post-processing modules on top of the fixed Plan A pipeline (YOLOv12-s, no architectural change), each targeting one of Plan A's two observed residual failure modes:

| Failure mode (from Plan A) | Phase 4 module |
|---|---|
| (a) False positives on rocks / huts / vegetation (precision-side errors) | **Hard-Negative Cascade Classifier** |
| (b) Bbox not tight enough at strict IoU (mAP@0.5:0.95 = 0.574 vs mAP@0.5 = 0.872) | **SAM2 Bbox Refinement** |

## Success criteria

| Metric | Plan A baseline | Phase 4 target | Stretch |
|---|---|---|---|
| mAP@0.5 | 0.872 | ≥ 0.880 | ≥ 0.890 |
| mAP@0.5:0.95 | 0.574 | ≥ 0.610 | ≥ 0.630 |
| mAP@0.75 | 0.661 | ≥ 0.700 | — |
| AP_small | 0.494 | ≥ 0.520 | — |
| Precision @ score 0.25 | (measure first) | +5% absolute | +10% absolute |

If Phase 4 lands at "target" → workshop paper. If only "stretch" lands on one of the two metrics → still publishable as ablation.

---

## Module 1 — SAM2 Bbox Refinement

### Idea

Plan A's SAHI output gives bboxes that are correctly located but often a few pixels loose around the person. Prompt SAM2 with each Plan A bbox → get a segmentation mask → refit a tight bbox from the mask. Push mAP@0.75 and mAP@0.5:0.95.

### Pipeline

```
Plan A predictions (bbox_i, score_i)
    │
    ▼  for each bbox_i:
SAM2.predict(image=tile, box_prompt=bbox_i)
    │
    ▼ mask_i
post-process mask:
  - keep largest connected component
  - if area_i < min_area or aspect_ratio out of [0.3, 4.0]: drop bbox_i (FP filter)
  - else: refit_bbox_i = tight_bbox(mask_i)
    │
    ▼
final predictions (refit_bbox_i, score_i)
```

### Implementation steps

1. **Install SAM2** (Meta AI). `pip install git+https://github.com/facebookresearch/sam2.git`. Download checkpoint `sam2_hiera_small.pt` (~46 M params) — small variant is enough for bbox prompting and runs on Kaggle T4.

2. **Wrap Plan A inference** to output `(image_path, [bbox, score])` tuples per val image — refactor existing SAHI inference script.

3. **Build SAM2 refinement wrapper:**
   ```python
   from sam2.build_sam import build_sam2
   from sam2.sam2_image_predictor import SAM2ImagePredictor

   predictor = SAM2ImagePredictor(build_sam2("sam2_hiera_s.yaml", "sam2_hiera_small.pt"))

   def refine(image, bbox):
       predictor.set_image(image)
       masks, scores, _ = predictor.predict(box=np.array(bbox), multimask_output=False)
       mask = masks[0]
       if mask.sum() < MIN_AREA:
           return None  # FP filter
       ys, xs = np.where(mask)
       refit = [xs.min(), ys.min(), xs.max(), ys.max()]
       ar = (refit[2]-refit[0]) / max(refit[3]-refit[1], 1)
       if ar < 0.3 or ar > 4.0:
           return None
       return refit
   ```

4. **Hyperparameter sweep:**
   - `MIN_AREA` ∈ {30, 50, 100, 200} px²
   - Aspect ratio bounds: try [0.2, 5.0] vs [0.3, 4.0] vs no bound
   - SAM2 image input: full 4000×3000 vs SAHI tile (640×640 with bbox in tile coords) — test both, native should be cleaner

5. **Evaluate** with pycocotools on HERIDAL val. Compare mAP@0.5, mAP@0.5:0.95, mAP@0.75, AP_small vs Plan A baseline.

### Expected outputs

- `results/phase4_sam2_refinement.json` — metrics with various hyperparams
- `results/visualizations/phase4_sam2/` — before/after grid for ~10 sample images

### Risks & mitigations

- **SAM2 may segment background instead of person at very small scales (<20 px).** Mitigation: fall back to original Plan A bbox if mask is degenerate.
- **SAM2 inference is slow** (~1–2 s per image with all bboxes). For 313 val images that's ~10 minutes — fine for evaluation, not for real-time. Document as "offline refinement, not real-time".
- **SAM2 may over-tighten bbox.** Compare against GT bbox tightness; consider 1–2 px outward padding after refit.

### Compute

~2 GPU-hours on Kaggle T4 for full eval + hyperparam sweep. No training.

---

## Module 2 — Hard-Negative Cascade Classifier

### Idea

Train a small image classifier on crops of Plan A's predictions, labelling each crop as `person` (true positive) or `not_person` (false positive on rocks / huts / vegetation). Apply at inference as a filter: drop any Plan A bbox whose classifier score for `not_person` is high.

### Data collection (the most important step)

1. Run Plan A inference on **HERIDAL train** (not val — val is for final eval). Output ~5,000 predictions.

2. Match each prediction against GT bboxes (IoU threshold 0.5):
   - IoU ≥ 0.5 → `person` (positive sample)
   - IoU < 0.1 → `not_person` (hard negative)
   - 0.1 ≤ IoU < 0.5 → ambiguous, drop

3. **Crop around each bbox** with padding (1.5× bbox size, resize to 64×64). This becomes the cascade classifier's training set.

4. Expected counts: ~3,000 positives, ~5,000–10,000 hard negatives (Plan A's FPs). If negatives < 5× positives, augment by sampling random non-person regions.

5. Save as `data/cascade/train/{person,not_person}/` — standard image classification layout.

### Model

- **Architecture:** MobileNetV3-Small or ResNet18 (ImageNet-pretrained). MobileNet is ~5 MB, ResNet18 ~45 MB.
- **Why small:** classifier runs inside the inference loop; must be fast. MobileNetV3-Small inference is < 1 ms per crop.
- **Training:** 30 epochs, AdamW, lr=1e-3, cosine schedule, binary cross-entropy. Standard `torchvision.models` + `timm` works.

### Integration with Plan A

```
Plan A predictions (bbox_i, score_i) on val image
    │
    ▼  for each bbox_i:
crop_i = crop_with_padding(image, bbox_i, scale=1.5, resize=64)
person_prob_i = cascade_classifier(crop_i)
    │
    ▼
if person_prob_i < THRESHOLD:
    drop bbox_i
else:
    keep with new score = score_i * person_prob_i
```

### Hyperparameter sweep

- THRESHOLD ∈ {0.3, 0.5, 0.7}
- Score combination: `score_i` (ignore classifier) vs `score_i * person_prob_i` (multiplicative) vs `0.5*score_i + 0.5*person_prob_i` (additive)
- Crop scale: 1.0× vs 1.5× vs 2.0× bbox

### Expected outputs

- `src/cascade/train_cascade.py` — training script
- `src/cascade/apply_cascade.py` — inference wrapper around Plan A
- `results/phase4_cascade.json` — metrics across hyperparams
- `results/cascade_confusion_matrix.png` — classifier-only eval on held-out val crops

### Risks & mitigations

- **Class imbalance** (negatives >> positives if Plan A's precision is high). Use weighted sampling or focal loss.
- **Domain shift between train/val negatives.** Mitigation: generate cascade train set from train+val *predictions* (not test), evaluate cascade itself on Plan A's val predictions excluded from training.
- **Cascade may drop true positives at small scale** (tiny persons look ambiguous). Mitigation: train classifier with heavy small-scale augmentation; consider scale-conditional threshold.

### Compute

- Data prep: 1 h on Kaggle T4 (Plan A inference on train + crop extraction)
- Classifier training: 1–2 h on Kaggle T4
- Eval sweep: 30 min

Total: **3–4 GPU-hours**.

---

## Module 3 — Combined Pipeline (Plan A + SAM2 + Cascade)

### Order of operations matters

Two reasonable orderings:

**Option A: SAM2 first, then Cascade**

```
Plan A → SAM2 refine (drop degenerate + tighten) → Cascade filter → final
```

Pros: Cascade sees cleaner bboxes (tightened by SAM2), better classification.
Cons: If SAM2 drops a TP, Cascade can't recover it.

**Option B: Cascade first, then SAM2**

```
Plan A → Cascade filter (drop FPs) → SAM2 refine (tighten survivors) → final
```

Pros: Less SAM2 work (only on survivors). Cascade decides on raw Plan A bbox.
Cons: Cascade sees looser bboxes — classification noisier.

→ **Test both orderings empirically.**

### Combined evaluation

| Variant | mAP@0.5 | mAP@0.5:0.95 | AP_small | Notes |
|---|---|---|---|---|
| Plan A (baseline) | 0.872 | 0.574 | 0.494 | reference |
| Plan A + SAM2 only | ? | ? | ? | mAP@0.75 should jump |
| Plan A + Cascade only | ? | ? | ? | Precision should jump |
| Plan A + Cascade → SAM2 | ? | ? | ? | full pipeline |
| Plan A + SAM2 → Cascade | ? | ? | ? | full pipeline alt |

This ablation table is the **core paper contribution** of Phase 4: showing each module's standalone gain + combined gain (with possible super-additivity if they fix orthogonal errors).

---

## Timeline (estimate, ~10 GPU-hours total)

| Day | Task | Compute |
|---|---|---|
| 1 | SAM2 install + wrapper + first eval | 2 h |
| 1 | SAM2 hyperparam sweep | 1 h |
| 2 | Cascade data collection (Plan A inference on train + crop extraction) | 1 h |
| 2 | Cascade training (MobileNetV3-Small) | 2 h |
| 3 | Cascade integration + sweep on val | 1 h |
| 3 | Combined pipeline ablation (both orderings) | 1 h |
| 4 | Visualizations + writeup | (no GPU) |
| 4 | Push to GitHub + update README | (no GPU) |

**All compute fits within Kaggle free tier (30 h/week).** No paid VM needed.

---

## Files to be added at the end of Phase 4

```
docs/
  phase4_results.md                       # writeup parallel to plan_a_analysis.md
src/
  cascade/
    train_cascade.py
    apply_cascade.py
  sam2_refine.py
  phase4_pipeline.py                      # combined eval entry point
results/
  phase4_sam2_refinement.json
  phase4_cascade.json
  phase4_combined.json
  cascade_confusion_matrix.png
  visualizations/
    phase4_sam2/                          # before/after grid for ~10 images
    phase4_cascade/                       # FP removal examples
    phase4_combined/                      # final pipeline samples
```

## Citations to add when Phase 4 lands

- **SAM 2:** Ravi, N., et al. (2024). *SAM 2: Segment Anything in Images and Videos.* [arXiv:2408.00714](https://arxiv.org/abs/2408.00714).
- **Cascade design:** Cai, Z., & Vasconcelos, N. (2018). *Cascade R-CNN: Delving into High Quality Object Detection.* CVPR. [arXiv:1712.00726](https://arxiv.org/abs/1712.00726).
- **MobileNetV3:** Howard, A., et al. (2019). *Searching for MobileNetV3.* ICCV. [arXiv:1905.02244](https://arxiv.org/abs/1905.02244).

## Fallback if Phase 4 underperforms

- If SAM2 alone gives < +0.01 mAP@0.5:0.95 → drop SAM2, keep Cascade only.
- If Cascade alone gives < +1% precision absolute → drop Cascade, keep SAM2 only.
- If both underperform → consider Module 4 (Multi-scale SAHI + WBF) as additional safety, paper as "ensemble baseline".
- Worst case: report Phase 4 as another negative result (Plan A is already a defensible workshop paper).
