# Progress Report — Weeks 1-3

**Project:** UAV Search & Rescue Person Detection with Synthetic Data Augmentation
**Reporting period:** Week 1 – Week 3
**Date:** 2026-05-13

---

## 1. Project Goal (recap)

Build a real-time UAV person detection pipeline for Search and Rescue (SAR), based on **YOLOv12** (NeurIPS 2025, [arXiv:2502.12524](https://arxiv.org/abs/2502.12524), [code](https://github.com/sunsmarterjie/yolov12)), and improve generalization by augmenting training data with synthetic imagery generated in the **AirSim** simulator ([arXiv:1705.05065](https://arxiv.org/abs/1705.05065), [code](https://github.com/microsoft/AirSim)). Final evaluation will include both real-image benchmarks and an end-to-end simulator-based demo.

The core research question:
> *"Can synthetic data from a UAV simulator measurably improve real-image person detection in SAR-relevant scenarios?"*

---

## 2. What Was Done

### Week 1 — Environment Setup

- Set up Kaggle, Weights & Biases, and Roboflow accounts
- Studied YOLOv12 paper (NeurIPS 2025) and Ultralytics framework
- Successfully ran smoke test: YOLOv12-n inference on a sample image (Kaggle T4 GPU)

### Week 2 — Baseline Training

**Dataset selection:** After evaluating multiple options (HERIDAL, TinyPerson, Roboflow Universe SAR datasets), selected **VisDrone-DET** ([Zhu et al., TPAMI 2021](https://arxiv.org/abs/2001.06303)) filtered to person classes only. Rationale: open access, large enough for training (5,684 train images), aerial perspective, and contains pedestrian/people classes relevant to SAR.

**Filtering pipeline:**
- Original VisDrone has 10 classes (pedestrian, car, van, ...). We keep only:
  - Class 0: pedestrian (79,337 instances)
  - Class 1: people (27,059 instances)
- Both remapped to a single class `person` (~106K instances total).

**Training configuration:**
- Model: YOLOv12-n (2.5M parameters, 6 GFLOPs)
- Image size: 640
- Batch size: 16
- Optimizer: AdamW, lr=0.001, cosine schedule
- Epochs: 80 (no early stopping triggered)
- Hardware: 1× Tesla T4 (Kaggle free tier)
- Training time: 2.25 hours

**Baseline results:**

| Metric | Value |
|---|---|
| mAP@0.5 | **0.476** |
| mAP@0.5:0.95 | 0.187 |
| Precision | 0.621 |
| Recall | 0.438 |
| Inference latency | 2.0 ms/image (~500 FPS on T4) |

**Observations:**
- mAP@0.5 of 0.476 is reasonable for a nano model on a hard urban-aerial dataset.
- Recall of 0.438 indicates that **~56% of person instances are missed**. This is mainly tiny persons in cluttered backgrounds — a known difficulty of VisDrone and a clear target for the synthetic-data experiments in Phase 3.
- Inference latency of 2.0 ms easily meets real-time UAV requirements (≥20 FPS).

### Week 3 — Size Comparison ✅ Done

Trained YOLOv12-s (~9.3M parameters, ~19 MB) on the same filtered dataset with identical recipe, in overnight commit-and-run mode (~3–4h on T4).

**Comparison:**

| Metric | YOLOv12-n | **YOLOv12-s** | Δ (s − n) |
|---|---|---|---|
| mAP@0.5 | 0.476 | **0.552** | **+0.076** |
| mAP@0.5:0.95 | 0.187 | 0.231 | +0.044 |
| Precision | 0.621 | 0.690 | +0.069 |
| Recall | 0.438 | 0.497 | +0.059 |

**Decision: YOLOv12-s is selected for the downstream synthetic-data experiments.**

Rationale:
- ΔmAP@0.5 = +0.076 exceeds the predefined 0.05 decision threshold.
- Recall improvement (+0.059) is particularly valuable for SAR, where missing a person is more costly than a false positive. YOLOv12-s misses roughly half of person instances; -n misses ~56%.
- Inference latency remains real-time-compatible (~5-8 ms estimated on T4; benchmarking in Week 4).
- Larger model is still deployable on Jetson Orin-class edge devices.

Full analysis: [`results/comparison_n_vs_s.md`](../results/comparison_n_vs_s.md).

---

## 3. Methodology Going Forward

### Phase 2 (Weeks 4-6) — Analysis & Simulator Setup

- Error analysis of baseline: identify where the model fails (tiny person, occlusion, vegetation) to motivate synthetic data design.
- Export model to ONNX, benchmark FPS for edge deployment.
- Install and configure AirSim on local machine.
- Build automated image-capture and labeling pipeline in AirSim.

### Phase 3 (Weeks 7-10) — Synthetic Data Experiments [core contribution]

- Generate ~1,500–2,000 synthetic SAR images with domain randomization (lighting, weather, vegetation, drone height, person pose).
- Train YOLOv12 on different mixtures of real + synthetic data.
- Ablations:
  - Synthetic ratio (0%, 25%, 50%, 100%, 200%)
  - Scenario contribution (forest only, mountain only, mixed)

### Phase 4 (Weeks 11-12) — Methodology Extension

Implement **one** of the following improvements (chosen based on Phase 3 results):
- **Curriculum learning:** pretrain on synthetic → fine-tune on real
- **Hard-case mining:** detect baseline failure modes → generate targeted synthetic samples

### Phase 5 (Week 13) — Cross-Domain Evaluation

- Evaluate models on a held-out SAR dataset (HERIDAL or SARD) to test generalization beyond VisDrone.
- Compare against alternative real-time detectors (RT-DETRv2 or YOLOv10) trained with the same recipe.

### Phase 6 (Weeks 14-16) — Demo + Writing

- End-to-end demo: drone autonomously searches for a person in AirSim, with real-time YOLOv12 detection. Measure time-to-find and success rate over multiple runs.
- Write a technical report and (optionally) a workshop-style paper.

---

## 4. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| AirSim installation issues on local machine | Backup: Gazebo + PX4 SITL, or Unity Perception |
| HERIDAL access takes too long (email-gated) | Use SARD or VisDrone test set as fallback for cross-evaluation |
| Synthetic data fails to improve metrics | Negative results are still publishable; pivot narrative to "quantifying the sim-to-real gap" |
| Kaggle session disconnects during long training | Adopt "Save & Run All (Commit)" mode for unattended overnight runs (already done in Week 3) |

---

## 5. Asks / Questions for Advisor

1. Is the choice of **VisDrone person subset** (urban-aerial) acceptable as a primary benchmark, given that HERIDAL (wilderness SAR) is access-gated? Or should we wait for HERIDAL access?
2. For the Week 11-12 methodology extension, do you have a preference between **curriculum learning** vs **hard-case mining**?
3. If results are strong, is the project a good candidate to submit to a regional venue (e.g., FAIR, RIVF, KSE) for student-level publication? Or do you suggest other venues?

---

## 6. HERIDAL Baseline (added 2026-05-14)

After pivoting the primary dataset from VisDrone (urban aerial) to HERIDAL (wilderness SAR), I retrained YOLOv12-s on HERIDAL with the same recipe (100 epochs, imgsz=640, batch=12, AdamW lr=0.001, Tesla T4 commit-and-run).

### Results

| Metric | VisDrone (urban) | **HERIDAL (wilderness)** | Δ |
|---|---|---|---|
| mAP@0.5 | 0.552 | **0.759** | **+0.207** |
| mAP@0.5:0.95 | 0.231 | **0.344** | **+0.113** |
| Precision | 0.690 | 0.749 | +0.059 |
| **Recall** | 0.497 | **0.713** | **+0.216** |

Despite HERIDAL having ~5× fewer training images, every metric improves substantially. Recall in particular jumps from 0.497 to 0.713, which is the most safety-critical metric for SAR (a missed person can mean a lost rescue).

Full analysis: [`results/comparison_visdrone_vs_heridal.md`](../results/comparison_visdrone_vs_heridal.md).

**HERIDAL is now the working baseline** for all downstream experiments. VisDrone becomes a cross-domain reference for Phase 5.

## 7. Advisor Feedback (received 2026-05-13, applied to HERIDAL)

The advisor reviewed the VisDrone baseline results (mAP@0.5 = 0.476 at that time) and proposed three improvement directions:

1. **Density-based clustering & cropping** of training images (split each image into two parts based on object density). The motivation is that wide-area aerial images have most pixels as empty background; clustering and cropping around dense regions concentrates training signal on the targets.
2. **Validate on HERIDAL** (now done — see Section 6).
3. **Address class imbalance (pedestrian 79K vs people 27K instances, ~80:20)** by generating synthetic data via AirSim.

### Adaptation for HERIDAL

Since HERIDAL has a single class (`person`), the 80:20 class imbalance does not directly apply. The underlying idea is reinterpreted as **generating underrepresented scenarios** in AirSim — occluded persons, lying poses, dense vegetation, and low-light conditions — rather than balancing two classes.

### Next steps based on feedback

- Phase 2.5 (next week): implement **SAHI** (Slicing Aided Hyper Inference) on the HERIDAL baseline first. This is an inference-time density-aware crop and requires no retraining, so it's a fast way to test the principle. If it helps, move to training-time density-aware augmentation.
- Phase 3 (weeks 6-9): use AirSim to generate ~1,500 synthetic SAR images focused on hard scenarios (occlusion, lying poses, low-light, dense forest). Mix with HERIDAL real data and measure mAP improvement.

## 8. Phase 2.5 — Density-aware training crops + SAHI (Plan A)

Following the advisor's suggestion (clustering and cropping images based on density), I implemented **density-aware crops at training time**, then evaluated with SAHI sliced inference on the original 4000×3000 validation images.

### Pipeline

1. From each 4000×3000 HERIDAL training image, generate ~4 crops of 640×640 at native resolution:
   - 2 crops centered on each ground-truth person (with random jitter)
   - 1 random crop per image (for background variety)
2. Re-train YOLOv12-s for 80 epochs on these ~4,500 crops (COCO pretrained init, AdamW, lr=0.001, AMP, T4 commit-and-run, ~3 hours).
3. Run SAHI sliced inference on the original 4000×3000 val set (no resize), 640×640 tiles with 0.2 overlap.
4. Evaluate with pycocotools COCO eval on the original-image coordinates.

### Why we did it this way

A first attempt applied SAHI inference directly to the baseline model (trained on resized 4000×3000 → 640 images). It failed catastrophically (mAP@0.5: 0.759 → 0.255) because of a training–inference distribution mismatch: the model had never been shown native-resolution patches and was hallucinating 354 detections on an image with one real person. Generating the crops at training time fixes this mismatch.

### Results

| Metric | Baseline | Naive SAHI ❌ | **Plan A** | Δ vs baseline |
|---|---|---|---|---|
| mAP@0.5 | 0.759 | 0.255 | **0.872** | **+0.113** |
| mAP@0.5:0.95 | 0.344 | 0.129 | **0.574** | **+0.230** |
| mAP@0.75 | — | 0.117 | **0.661** | (large) |
| AP_small | — | 0.002 | **0.494** | (~250× over naive SAHI) |
| AR_100 | — | 0.438 | **0.662** | +0.224 |

Full analysis: [`results/plan_a_analysis.md`](../results/plan_a_analysis.md).

### Implications

- The advisor's density-crop suggestion is **strongly validated**: +66% relative on mAP@0.5:0.95, and AP_small (the tiny-person metric) is now usable (0.494) where it was effectively zero before.
- The earlier mAP@0.5:0.95 weakness is largely resolved.
- The training-time crop pipeline can be reused to mix in synthetic crops during Phase 3 (AirSim).
- Naive SAHI inference is now a documented negative result and serves as a useful contrast in the eventual report.

## 9. Summary

Phases 1, 2, and 2.5 are complete. The current working baseline on HERIDAL is **Plan A: mAP@0.5 = 0.872, mAP@0.5:0.95 = 0.574, AP_small = 0.494**. The next focus is Phase 3 (AirSim synthetic data) — generating ~1,500 simulator-rendered SAR images for hard scenarios (occlusion, lying poses, dense vegetation, low-light), processed through the same crop pipeline and mixed into training.

---

## References

- **YOLOv12** (base method) — Tian, Y., Ye, Q., & Doermann, D. (2025). *Attention-Centric Real-Time Object Detectors.* NeurIPS 2025. [arXiv:2502.12524](https://arxiv.org/abs/2502.12524) · [code](https://github.com/sunsmarterjie/yolov12)
- **VisDrone** (dataset) — Zhu, P. et al. (2021). *Detection and Tracking Meet Drones Challenge.* TPAMI. [arXiv:2001.06303](https://arxiv.org/abs/2001.06303) · [dataset](https://github.com/VisDrone/VisDrone-Dataset)
- **AirSim** (simulator) — Shah, S. et al. (2017). *High-Fidelity Visual and Physical Simulation for Autonomous Vehicles.* FSR. [arXiv:1705.05065](https://arxiv.org/abs/1705.05065) · [code](https://github.com/microsoft/AirSim)

---

*Repo:* https://github.com/Lucas-on-the-cloud/UAV_res
