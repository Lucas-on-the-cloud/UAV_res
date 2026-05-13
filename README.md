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
| 1. Setup & Baseline | 1-3 | ✅ Done |
| 2. Error Analysis & Simulator | 4-6 | ⚪ Upcoming |
| 3. Synthetic Data Experiments | 7-10 | ⚪ Upcoming |
| 4. Methodology Extension | 11-12 | ⚪ Upcoming |
| 5. Cross-Dataset Evaluation | 13 | ⚪ Upcoming |
| 6. Demo & Writing | 14-16 | ⚪ Upcoming |

## Key Results (Weeks 2-3)

Baseline on VisDrone person-filtered (5,684 train / 531 val, 80 epochs, T4):

| Metric | YOLOv12-n | **YOLOv12-s** *(selected)* |
|---|---|---|
| mAP@0.5 | 0.476 | **0.552** |
| mAP@0.5:0.95 | 0.187 | 0.231 |
| Precision | 0.621 | 0.690 |
| Recall | 0.438 | 0.497 |
| Params (M) | 2.5 | 9.3 |
| Best.pt size (MB) | 5.4 | ~19 |

**YOLOv12-s selected for downstream experiments** (ΔmAP@0.5 = +0.076 ≥ 0.05 threshold).
See [results/comparison_n_vs_s.md](results/comparison_n_vs_s.md) for full analysis.

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
