# UAV Search & Rescue Person Detection

A research project on **real-time UAV-based person detection for Search and Rescue (SAR)** scenarios, combining state-of-the-art object detection with synthetic data augmentation from a drone simulator.

## Overview

**Base method:** YOLOv12 — Attention-Centric Real-Time Object Detectors (NeurIPS 2025)
&nbsp;&nbsp;&nbsp;&nbsp;📄 Paper: [arXiv:2502.12524](https://arxiv.org/abs/2502.12524)
&nbsp;&nbsp;&nbsp;&nbsp;💻 Code: [sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12)

**Dataset:** VisDrone-DET (person-filtered)
&nbsp;&nbsp;&nbsp;&nbsp;📄 Paper: [Detection and Tracking Meet Drones Challenge](https://arxiv.org/abs/2001.06303) (TPAMI 2021)
&nbsp;&nbsp;&nbsp;&nbsp;🔗 Dataset: [github.com/VisDrone](https://github.com/VisDrone/VisDrone-Dataset) · Kaggle mirror: `banuprasadb/visdrone-dataset`

**Simulator:** AirSim
&nbsp;&nbsp;&nbsp;&nbsp;📄 Paper: [AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles](https://arxiv.org/abs/1705.05065) (FSR 2017)
&nbsp;&nbsp;&nbsp;&nbsp;💻 Code: [Microsoft/AirSim](https://github.com/microsoft/AirSim)

**Application:** UAV-mounted person detection for SAR
**Novelty:** Synthetic data augmentation generated in AirSim simulator to improve generalization
**Evaluation:** Real drone imagery (VisDrone person subset; later HERIDAL/SARD for cross-domain) + end-to-end simulator demo

## Current Status

| Phase | Weeks | Status |
|---|---|---|
| 1. Setup & VisDrone baseline | 1-3 | ✅ Done |
| 2. HERIDAL baseline | 4 | ✅ Done |
| 2.5. Density-based cropping (SAHI) | 5 | 🟢 Next |
| 3. AirSim synthetic data | 6-9 | ⚪ Upcoming |
| 4. Methodology extension | 10-11 | ⚪ Upcoming |
| 5. Cross-domain eval | 12 | ⚪ Upcoming |
| 6. Demo & writing | 13-16 | ⚪ Upcoming |

## Key Results

### Phase 1 — VisDrone baseline (urban aerial, 5,684 train images, 80 epochs)

| Metric | YOLOv12-n | YOLOv12-s |
|---|---|---|
| mAP@0.5 | 0.476 | 0.552 |
| mAP@0.5:0.95 | 0.187 | 0.231 |
| Recall | 0.438 | 0.497 |

YOLOv12-s selected (ΔmAP@0.5 = +0.076). See [comparison_n_vs_s.md](results/comparison_n_vs_s.md).

### Phase 2 — HERIDAL baseline (wilderness SAR, 1,124 train images, 100 epochs) ⭐

| Metric | VisDrone (urban) | **HERIDAL (wilderness)** | Δ |
|---|---|---|---|
| mAP@0.5 | 0.552 | **0.759** | **+0.207** |
| mAP@0.5:0.95 | 0.231 | **0.344** | **+0.113** |
| Precision | 0.690 | 0.749 | +0.059 |
| **Recall** | 0.497 | **0.713** | **+0.216** |

HERIDAL substantially outperforms VisDrone despite having 5× fewer training images, confirming domain alignment matters more than dataset size for the SAR application. Detailed analysis: [comparison_visdrone_vs_heridal.md](results/comparison_visdrone_vs_heridal.md).

**HERIDAL is now the working baseline** for all downstream experiments. VisDrone becomes a cross-domain reference.

## Repository Structure

```
UAV_res/
├── README.md                  # This file
├── docs/
│   ├── proposal.md            # 16-week project proposal
│   ├── progress_report.md     # Weekly progress (English)
│   └── dataset_selection.md   # Notes on dataset choice
├── notebooks/
│   └── (Kaggle notebook links + exported .ipynb)
├── src/
│   ├── filter_visdrone.py     # Filter VisDrone to person-only
│   └── convert_voc_to_yolo.py # Generic VOC/CSV → YOLO converter
├── configs/
│   └── visdrone_person.yaml   # Dataset config for YOLOv12 training
├── results/
│   └── yolov12n_baseline_summary.json
└── .gitignore
```

## Compute

- Training: Kaggle free tier (2× Tesla T4, 30h/week)
- Simulator: Local PC + AirSim
- Total estimated compute: ~90 hours T4 (within free quota)

## Tech Stack

- PyTorch 2.10 + CUDA 12.8
- Ultralytics 8.4.x
- YOLOv12 (Ultralytics-compatible weights)
- AirSim (Microsoft)
- Weights & Biases for experiment tracking

## References

### Base method
- Tian, Y., Ye, Q., & Doermann, D. (2025). **YOLOv12: Attention-Centric Real-Time Object Detectors.** NeurIPS 2025.
  [arXiv:2502.12524](https://arxiv.org/abs/2502.12524) · [Code](https://github.com/sunsmarterjie/yolov12)

### Dataset
- Zhu, P., Wen, L., Du, D., Bian, X., Fan, H., Hu, Q., & Ling, H. (2021). **Detection and Tracking Meet Drones Challenge.** TPAMI.
  [arXiv:2001.06303](https://arxiv.org/abs/2001.06303) · [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)

### Simulator
- Shah, S., Dey, D., Lovett, C., & Kapoor, A. (2017). **AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles.** FSR.
  [arXiv:1705.05065](https://arxiv.org/abs/1705.05065) · [Code](https://github.com/microsoft/AirSim)

### Related work referenced in the proposal
- Tobin, J. et al. (2017). **Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World.** IROS. [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)
- Akyon, F. C., Altinuc, S. O., & Temizel, A. (2022). **Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection (SAHI).** ICIP. [arXiv:2202.06934](https://arxiv.org/abs/2202.06934)
- Božić-Štulić, D., Marušić, Ž., & Gotovac, S. (2019). **Deep Learning Approach in Aerial Imagery for Supporting Land Search and Rescue Missions.** IJCV. [DOI](https://doi.org/10.1007/s11263-019-01177-1) · *HERIDAL dataset paper.*

## License

Research/academic use. See individual paper/dataset licenses for upstream materials.
