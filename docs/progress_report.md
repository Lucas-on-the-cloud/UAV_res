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

## 6. Summary

Phase 1 (Weeks 1-3) is **complete**. After comparing two model sizes, **YOLOv12-s** was selected for downstream experiments with baseline **mAP@0.5 = 0.552, recall = 0.497** on VisDrone person-filtered. Recall remains the main target for improvement, motivating the synthetic-data experiments planned for Phase 3.

Next major milestone: completing AirSim setup and the synthetic data pipeline by Week 6, with the first mixed-training experiment in Week 8.

---

## References

- **YOLOv12** (base method) — Tian, Y., Ye, Q., & Doermann, D. (2025). *Attention-Centric Real-Time Object Detectors.* NeurIPS 2025. [arXiv:2502.12524](https://arxiv.org/abs/2502.12524) · [code](https://github.com/sunsmarterjie/yolov12)
- **VisDrone** (dataset) — Zhu, P. et al. (2021). *Detection and Tracking Meet Drones Challenge.* TPAMI. [arXiv:2001.06303](https://arxiv.org/abs/2001.06303) · [dataset](https://github.com/VisDrone/VisDrone-Dataset)
- **AirSim** (simulator) — Shah, S. et al. (2017). *High-Fidelity Visual and Physical Simulation for Autonomous Vehicles.* FSR. [arXiv:1705.05065](https://arxiv.org/abs/1705.05065) · [code](https://github.com/microsoft/AirSim)

---

*Repo:* https://github.com/Lucas-on-the-cloud/UAV_res
