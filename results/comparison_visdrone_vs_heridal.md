# Cross-Domain Comparison — VisDrone vs HERIDAL

**Model:** YOLOv12-s (same architecture, same training recipe)
**Training:** 80–100 epochs, AdamW lr=0.001, cosine LR, AMP, Tesla T4

## Results

| Metric | VisDrone (urban aerial) | **HERIDAL (wilderness SAR)** | Δ (HERIDAL − VisDrone) |
|---|---|---|---|
| mAP@0.5 | 0.552 | **0.759** | **+0.207** |
| mAP@0.5:0.95 | 0.231 | **0.344** | **+0.113** |
| Precision | 0.690 | 0.749 | +0.059 |
| **Recall** | 0.497 | **0.713** | **+0.216** |
| Train images | 5,684 | 1,124 | −4,560 |
| Val images | 531 | 313 | −218 |
| Classes | person (merged ped+people) | person | 1 vs 1 |

## Analysis

### Why HERIDAL outperforms VisDrone

Despite having ~5× fewer training images, the YOLOv12-s model performs substantially better on HERIDAL across all metrics. Several factors contribute:

1. **Single-class detection.** HERIDAL contains only `person`, whereas our VisDrone filtered set merges `pedestrian` and `people` classes. Even after merging, the original distribution differences induce internal label noise that hurts metrics.

2. **Cleaner backgrounds.** Wilderness scenes (forests, rocky terrain) have far fewer distractors than urban scenes (vehicles, signs, traffic). Background reduces false-positive load and lets the model concentrate on person features.

3. **Higher-quality annotations.** HERIDAL is an academic SAR benchmark; bounding boxes are professionally annotated.

4. **Domain alignment.** HERIDAL matches the project's actual research goal (wilderness search and rescue). The high recall (0.713) means very few persons are missed — critical for SAR.

### Implications

- **HERIDAL is the right primary benchmark** for this project. VisDrone now serves as a cross-domain reference, not the main evaluation.
- The recall improvement (0.497 → 0.713) directly addresses the main weakness identified in the VisDrone baseline.
- However, **mAP@0.5:0.95 is still relatively low (0.344)**, indicating that bounding-box localization for tiny persons remains imprecise. This is the next target for improvement (see professor's feedback re: density-based cropping).

## Selected baseline for downstream

**YOLOv12-s on HERIDAL** (mAP@0.5 = 0.759, recall = 0.713) is now the working baseline for:

- Phase 2.5: density-based cropping experiments (target mAP@0.5:0.95 improvement)
- Phase 3: AirSim synthetic data augmentation focused on hard SAR scenarios
- Phase 5: cross-domain evaluation on VisDrone (already trained, no extra compute needed)

## Files

- Baseline weights: `yolov12s_heridal_BEST.pt` (stored locally + Google Drive)
- Metrics: [`yolov12s_heridal_baseline_summary.json`](yolov12s_heridal_baseline_summary.json)
- Training notebook: `heridal-trainning-yolov12s` (Kaggle)
