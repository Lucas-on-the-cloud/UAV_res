# Model Size Comparison — YOLOv12-n vs YOLOv12-s

**Dataset:** VisDrone person-filtered (5,684 train / 531 val, 106K person instances)
**Training:** 80 epochs, AdamW lr=0.001, cosine LR, AMP, Tesla T4

## Results

| Metric | YOLOv12-n | YOLOv12-s | Δ (s − n) |
|---|---|---|---|
| **mAP@0.5** | 0.476 | **0.552** | **+0.076** |
| mAP@0.5:0.95 | 0.187 | 0.231 | +0.044 |
| Precision | 0.621 | 0.690 | +0.069 |
| Recall | 0.438 | 0.497 | +0.059 |
| Params (M) | 2.5 | 9.3 | 3.7× |
| Best.pt size (MB) | 5.4 | ~19 | 3.5× |
| Training time (h on T4) | 2.25 | ~3-4 | ~1.5× |
| Inference latency (ms on T4) | ~2.0 | ~5-8 (estimated) | ~3-4× |

## Decision

**Selected model for downstream experiments (Phases 3-6): YOLOv12-s**

### Rationale

- ΔmAP@0.5 of +0.076 exceeds the predefined 0.05 threshold ⇒ size upgrade is justified.
- **Recall improved by +0.059** (0.438 → 0.497). Recall is the more critical metric for Search-and-Rescue applications, where missing a person has higher cost than a false alarm. The nano model misses ~56% of person instances; the small model misses ~50%.
- Precision also improved by +0.069, indicating better discrimination, not just more aggressive predictions.
- Inference latency is still well within real-time UAV constraints (~5-8 ms expected ⇒ >100 FPS on T4; should remain real-time on Jetson Orin).

### Trade-off considerations

- Model size (~19 MB) is larger but still deployable on edge devices.
- Training is ~1.5× slower per run — acceptable for the ablations planned in Phase 3.
- For very constrained edge hardware (Jetson Nano, Coral TPU), YOLOv12-n could be kept as a deployment fallback while -s remains the research target.

## Implications for the project narrative

The improvement when scaling from nano to small (with identical training recipe and data) reflects the inherent capacity gap between model sizes — it is not yet the *novelty* of this project. The intended novelty (synthetic data augmentation, Phase 3) will be evaluated **on top of YOLOv12-s** to give the strongest possible baseline.
