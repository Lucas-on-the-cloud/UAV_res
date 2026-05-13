# Project Proposal — 16-Week Plan

**Project:** UAV Search & Rescue Person Detection with Synthetic Data Augmentation
**Base paper:** YOLOv12 — Attention-Centric Real-Time Object Detectors (NeurIPS 2025)
**Author:** [Your name]
**Advisor:** [Advisor name]
**Created:** 2026-05-07
**Last updated:** 2026-05-13

---

## 1. Motivation

UAV-based person detection has critical real-world applications in Search and Rescue (SAR) — finding missing hikers, drowning victims, or people trapped in disaster zones. Two key technical challenges:

1. **Tiny object detection:** People photographed from drone altitude (30-150m) occupy very few pixels, making them hard to detect reliably.
2. **Data scarcity:** Real SAR imagery is difficult to collect (privacy, ethics, rarity of events), limiting supervised training.

We address these challenges by:
- Applying a state-of-the-art real-time detector (YOLOv12, NeurIPS 2025) to the SAR domain
- Augmenting training with synthetic data generated in the AirSim drone simulator, leveraging domain randomization

---

## 2. Research Questions

- **Q1.** Does synthetic data from a UAV simulator measurably improve detection accuracy on real SAR images?
- **Q2.** What is the optimal mixing ratio of real vs synthetic data?
- **Q3.** Which simulator scenario types (forest, mountain, field) contribute most?
- **Q4.** Can a curriculum (pretrain-on-sim → fine-tune-on-real) outperform joint mixed training?

---

## 3. Project Plan (16 weeks)

### Phase 1 — Setup & Baseline (Weeks 1-3)

- **W1:** Environment setup (Kaggle, W&B, Roboflow); smoke test YOLOv12
- **W2:** Dataset preparation (VisDrone person filter); train YOLOv12-n baseline
- **W3:** Train YOLOv12-s; n-vs-s comparison; pick model size for downstream

### Phase 2 — Analysis & Simulator (Weeks 4-6)

- **W4:** Error analysis of baseline; export ONNX; FPS benchmarking
- **W5:** AirSim installation; environment selection (forest, mountain)
- **W6:** Automated image-capture & labeling pipeline in AirSim

### Phase 3 — Synthetic Data Experiments (Weeks 7-10) [core contribution]

- **W7:** Generate ~1,500–2,000 synthetic images with domain randomization
- **W8:** First mixed-training experiment (real + 50% synthetic)
- **W9:** Ratio ablation (0%, 25%, 50%, 100%, 200%)
- **W10:** Scenario ablation (forest only / mountain only / mixed)

### Phase 4 — Methodology Extension (Weeks 11-12)

Choose **one** based on Phase 3 results:
- **W11-12 (option A):** Curriculum learning — pretrain on sim, fine-tune on real
- **W11-12 (option B):** Hard-case mining — generate targeted synthetic samples for baseline failure modes

### Phase 5 — Cross-Dataset & Baseline Compare (Week 13)

- Cross-evaluation on HERIDAL or SARD (if accessible)
- Compare against RT-DETRv2 or YOLOv10 trained with same recipe

### Phase 6 — Demo, Writing, Submission (Weeks 14-16)

- **W14:** End-to-end simulator demo (drone search with real-time inference)
- **W15:** Technical report (15-20 pages) + workshop paper draft (8 pages)
- **W16:** Polish + submit (target: FAIR, RIVF, KSE, or international workshop)

---

## 4. Datasets

| Dataset | Role | Access |
|---|---|---|
| **VisDrone-DET** (person-filtered) | Primary training/eval | Public (Kaggle mirror) |
| **HERIDAL** | Cross-domain eval (optional) | Email request |
| **SARD** | Cross-domain backup | Public or Kaggle |
| **AirSim-generated synthetic** | Augmentation | Generated in Phase 2-3 |

---

## 5. Evaluation Metrics

- **Detection:** mAP@0.5, mAP@0.5:0.95, AP_small, Precision, Recall
- **Real-time:** Inference latency (ms), FPS on Tesla T4 (proxy for Jetson)
- **End-to-end (simulator):** Time-to-find, success rate over N=20 runs, false alarm rate

---

## 6. Compute Resources

- **Primary:** Kaggle free tier (2× Tesla T4, 30h/week) → ~120h available across 4 weeks if needed
- **Estimated total need:** ~90h T4
- **Local:** AirSim runs on local PC (needs ~GTX 1060+ for photo-realistic mode)
- **No paid cloud compute anticipated**

---

## 7. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| AirSim install failure | Switch to Gazebo + PX4 SITL or Unity Perception |
| HERIDAL access delayed | Use SARD or VisDrone test set |
| Synthetic data does not improve metrics | Reframe as "quantifying the sim-to-real gap" — negative results are publishable |
| Kaggle disconnections during long training | Use "Save & Run All (Commit)" overnight |
| Insufficient time for Phase 6 writing | Cut Phase 4 to a single experiment or skip Phase 5 baselines |

---

## 8. Submission Targets (Phase 6)

### Regional (primary)

- **FAIR** (Vietnamese workshop, June-July)
- **RIVF** (IEEE Vietnam, March-April)
- **KSE** (IEEE Vietnam, June-August)
- **NICS** (IEEE Vietnam, Sept-Oct)

### International workshops (stretch)

- CVPR-W EarthVision
- ICCV-W Vision for All Seasons
- IROS / ICRA workshops on aerial robotics

---

## 9. Expected Outcomes

- A trained, real-time UAV person detection model
- A reproducible synthetic-data generation pipeline using AirSim
- Quantitative analysis of synthetic vs real data trade-offs in this domain
- An end-to-end demo (video) of drone-based SAR search in simulation
- A technical report and (optionally) a submitted paper

---

## 10. References (selected)

- Tian, Y., Ye, Q., & Doermann, D. (2025). **YOLOv12: Attention-Centric Real-Time Object Detectors.** NeurIPS 2025. [arXiv:2502.12524](https://arxiv.org/abs/2502.12524) · [code](https://github.com/sunsmarterjie/yolov12)
- Shah, S. et al. (2017). **AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles.** FSR. [arXiv:1705.05065](https://arxiv.org/abs/1705.05065) · [code](https://github.com/microsoft/AirSim)
- Tobin, J. et al. (2017). **Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World.** IROS. [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)
- Akyon, F. C. et al. (2022). **SAHI: Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection.** ICIP. [arXiv:2202.06934](https://arxiv.org/abs/2202.06934)
- Zhu, P. et al. (2021). **Detection and Tracking Meet Drones Challenge (VisDrone).** TPAMI. [arXiv:2001.06303](https://arxiv.org/abs/2001.06303) · [dataset](https://github.com/VisDrone/VisDrone-Dataset)
- Božić-Štulić, D., Marušić, Ž., & Gotovac, S. (2019). **Deep Learning Approach in Aerial Imagery for Supporting Land Search and Rescue Missions.** IJCV. [DOI](https://doi.org/10.1007/s11263-019-01177-1) — *HERIDAL dataset.*
