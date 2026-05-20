# 10-Minute Presentation Script — UAV SAR Person Detection

**Audience:** 3 people (advisor + reviewers)
**Length target:** 10 minutes, ~1700 spoken words
**Tone:** formal but conversational; honest about negative result

---

## [SLIDE 1 — Title]

*(Speak ~20 seconds)*

Good morning. Today I'm presenting my work on **real-time UAV-based person detection for wilderness Search and Rescue**. The main result I'll discuss is a pipeline built on top of YOLOv12-s that improves mean Average Precision on the HERIDAL dataset from **0.759 to 0.872** — a fifteen percent relative improvement — without modifying the detector architecture itself.

I'll cover the problem, the methodology in three layers, one negative result that I want to be transparent about, and the planned next phase.

---

## [SLIDE 2 — The problem]

*(Speak ~45 seconds)*

When a person goes missing in wilderness — a lost hiker, a stranded climber, survivors after a flood or earthquake — the search radius grows quadratically with elapsed time. Ground search teams cover terrain slowly. A drone can sweep the same area orders of magnitude faster, but a human operator watching a live video feed cannot reliably spot a thirty-pixel-wide person against forest or rocks over hours of footage.

This is the bottleneck I'm trying to address: **turn the drone's camera feed into actionable detections, automatically and in real time.**

The technical problem reduces to three coupled difficulties. First, persons are **tiny** — typically about thirty pixels wide in a four-thousand by three-thousand pixel aerial image, which is less than one one-thousandth of the image area. Second, the wilderness contains **natural distractors** that share person-like silhouettes — rocks, huts, dense vegetation. Third, labelled SAR data is **scarce**: the HERIDAL dataset I use has only 1,124 training images, two orders of magnitude smaller than COCO.

These three constraints drive every design decision in the project.

---

## [SLIDE 3 — Approach: a pipeline, not a new detector]

*(Speak ~45 seconds)*

The project deliberately does **not** propose a new detector architecture. The base detector — **YOLOv12-small**, from the NeurIPS 2025 paper by Tian and colleagues — is **held constant across every phase** of the work. Same architecture, same backbone, same released weights initialization.

This is a methodological choice. The research question is *"how can we improve aerial SAR person detection through data and pipeline design?"*, not *"can we design a better detector?"*. Holding the detector fixed makes every improvement I report cleanly attributable to a specific outside-the-model change. Anyone can swap YOLOv12 for YOLOv11 or RT-DETR and re-run the pipeline to verify the contribution generalizes.

The pipeline is structured in three layers, applied cumulatively: **dataset selection**, **train–inference scale alignment**, and **post-processing**.

---

## [SLIDE 4 — Layer 1: Dataset selection (VisDrone → HERIDAL)]

*(Speak ~50 seconds)*

I train on two aerial datasets in sequence. VisDrone, the urban-aerial benchmark, comes first — not because urban scenes match SAR, but because VisDrone serves three concrete methodological purposes. It validates that the YOLOv12 training and evaluation pipeline works end-to-end on a well-known benchmark. It lets me ablate the YOLOv12 variant size — nano versus small — at sufficient statistical power; small wins by 0.076 mAP@0.5. And it provides a reference number against the broader community.

Then I switch to **HERIDAL**, the actual target. HERIDAL is wilderness aerial imagery with high-altitude, near-top-down views, exactly the deployment scenario for SAR.

On HERIDAL with the same YOLOv12-s configuration, mAP@0.5 reaches **0.759** — substantially higher than VisDrone's 0.552 despite five times less data, because wilderness backgrounds are semantically simpler than urban ones.

So far so good. But there's a localization problem.

---

## [SLIDE 5 — The localization problem and naive SAHI failure]

*(Speak ~55 seconds)*

The Phase 2 baseline has mAP@0.5 = 0.759 but mAP@0.5:0.95 = only **0.344**. Reading those two metrics together: the model **finds** persons reliably at IoU threshold 0.5, but does **not** localize them tightly at stricter thresholds. The bounding boxes are loose.

The reason is a resolution mismatch. HERIDAL images are four-thousand by three-thousand pixels. YOLOv12 trains at `imgsz=640`, which means the trainer resizes each image to 640 by 640, shrinking a thirty-pixel person to roughly **five pixels**. The model never sees what a person looks like at its actual scale.

A natural remedy is **SAHI** — Slicing Aided Hyper Inference — from Akyon and colleagues at ICIP 2022. SAHI slices the original four-thousand by three-thousand image into 640-pixel tiles at native resolution at inference time, so persons appear at their real thirty-pixel scale.

I tried this. It **failed catastrophically**: mAP@0.5 dropped from 0.759 to 0.255. AP_small collapsed to essentially zero. The model hallucinated hundreds of detections per tile.

The diagnosis is a training–inference distribution mismatch. The model only ever saw heavily downsampled images during training. When SAHI fed it native-resolution patches at inference, the input was completely out-of-distribution. SAHI alone, applied as a post-hoc trick, is not enough.

---

## [SLIDE 6 — Layer 2: Plan A — density-aware crops + SAHI ⭐]

*(Speak ~70 seconds — this is the main contribution slide, slow down a bit)*

The fix — which I call **Plan A**, and which is the main contribution of the project — is to move the density-aware step **into training**, so that training and inference distributions match.

The recipe is three steps. **Step one**: from each four-thousand by three-thousand training image, generate native-resolution 640-by-640 crops. Two of these are person-centered with random offset, and one is a random crop. This produces about 4,500 training crops, all at native pixel scale, no resize. **Step two**: fine-tune YOLOv12-s on these crops for 80 epochs, same as baseline. **Step three**: at inference time, run SAHI sliced inference on the original 4000-by-3000 validation images.

Now the training distribution — native-resolution 640 crops — matches the inference distribution — native-resolution 640 SAHI tiles. The detector has seen what persons look like at their actual scale.

The results: **mAP@0.5 jumps from 0.759 to 0.872**, a fifteen percent relative improvement. **mAP@0.5:0.95 jumps from 0.344 to 0.574**, a sixty-six percent relative improvement — the localization problem is largely resolved. And the metric I think is most important, **AP_small** — the COCO small-object AP — jumps from essentially zero under naive SAHI to **0.494**, a roughly two-hundred-fifty-fold improvement.

The key insight, which I want to highlight because it generalizes beyond this project: **density-aware inference only works when the model has also seen density-aware training samples**. The two have to be coupled. SAHI as a standalone inference-time technique is insufficient when training resolution doesn't match.

---

## [SLIDE 7 — Phase 3: an honest negative result]

*(Speak ~60 seconds — be straightforward about the failure)*

I want to be transparent about Phase 3, which was an attempt to add synthetic data augmentation on top of Plan A. The original plan used the AirSim drone simulator, but AirSim had unresolvable dependency issues on modern Python, so I pivoted to **Stable Diffusion XL** for generating synthetic aerial backgrounds.

Two variants were attempted, both with rented GPU compute. The first generated two thousand SDXL backgrounds and composited HERIDAL person crops onto them. The second used same-domain HERIDAL-to-HERIDAL copy-paste compositing — no domain gap by construction.

Both **collapsed to essentially zero mAP**. Not regressions — collapses.

The root-cause diagnosis is again a training–inference scale mismatch — the same problem Plan A had solved. Both Phase 3 variants mixed synthetic data with **raw four-thousand by three-thousand HERIDAL images** during training, at `imgsz=640`. This re-introduced the five-pixel-person training distribution that Plan A had specifically removed. The synthetic samples themselves were fine; the pipeline that consumed them was wrong.

Plan A weights are unchanged, mAP@0.5 = 0.872 stands. Phase 3 is documented as a negative result, with the specific lesson — *to benefit from synthetic augmentation under SAHI, synthetic samples must go through the same density-aware crop pipeline as the real data* — recorded as "crop-then-synthesize" future work.

I'm reporting this rather than burying it because the lesson is reusable.

---

## [SLIDE 8 — Layer 3: Phase 4 — post-processing (in progress)]

*(Speak ~45 seconds)*

Phase 4, currently executing on Kaggle GPUs as I speak, adds two complementary post-processing modules on top of Plan A. Neither retrains YOLOv12.

The first module is a **hard-negative cascade classifier**. I train a small MobileNetV3 on crops of Plan A's predictions, labelled by IoU against ground truth — true positives versus false positives on rocks, huts, vegetation. At inference, the cascade filters Plan A's predictions, targeting the precision side.

The second module is **SAM2 bounding-box refinement**, from Meta AI's 2024 paper. Each Plan A bbox is used as a box prompt to SAM2, which returns a segmentation mask. I refit a tight bounding box from the mask, which directly raises mAP at the strict IoU thresholds.

Both modules target a different residual failure mode of Plan A — the cascade targets false positives, SAM2 targets loose boxes — so they should be additive in a combined pipeline. That combined ablation is the planned Phase 4 contribution.

---

## [SLIDE 9 — Summary contribution stack]

*(Speak ~40 seconds)*

To summarize the contribution stack as it stands today: on a fixed YOLOv12-s base, Phase 1 establishes the urban-aerial reference number on VisDrone at 0.552 mAP@0.5. Phase 2 switches to HERIDAL — the actual SAR target — reaching 0.759. Phase 2.5, Plan A, adds density-aware training crops and SAHI sliced inference, reaching **0.872 mAP@0.5 and 0.574 mAP@0.5:0.95**, with AP_small at 0.494.

Phase 3 is a documented negative result. Phase 4, in progress, layers post-processing.

Every reported delta attaches to a specific outside-the-model component — dataset, preprocessing, inference strategy, or post-processing. The detector architecture has not been changed at any point.

---

## [SLIDE 10 — Closing]

*(Speak ~30 seconds)*

The takeaways I'd like to leave with the committee. **First**: in aerial small-object detection, training–inference distribution alignment is the dominant lever, more so than architectural changes. **Second**: SAHI-style sliced inference is only effective when paired with native-resolution training data. **Third**: synthetic-data augmentation in this regime is non-trivial and requires the same scale-alignment discipline.

All artifacts, weights, code, and the full per-phase write-up are on the project GitHub repository. I'm happy to take questions.

---

## Timing breakdown

| Slide | Topic | Time |
|---|---|---|
| 1 | Title | 0:20 |
| 2 | Problem | 0:45 |
| 3 | Approach | 0:45 |
| 4 | Layer 1 datasets | 0:50 |
| 5 | Localization + naive SAHI | 0:55 |
| 6 | **Plan A (main)** | 1:10 |
| 7 | Phase 3 negative result | 1:00 |
| 8 | Phase 4 plan | 0:45 |
| 9 | Summary | 0:40 |
| 10 | Closing | 0:30 |
| **Total** | | **~8:20** |

Leaves ~1:40 buffer for slide transitions, breathing, and one or two casual asides. Don't memorize verbatim — internalize each slide's *one main point* and improvise around it. Slow down on slides 6 and 7 (Plan A + negative result) — those are the parts the committee will probe in Q&A.

## Anticipated Q&A — prepare 1-sentence answers

1. **"Why YOLOv12 specifically?"** — NeurIPS 2025, real-time inference (1.6 ms on T4), Ultralytics-compatible install. The contribution is data/pipeline-side and doesn't depend on this choice.
2. **"Why not just train at higher imgsz?"** — Tried; memory cost is quadratic and persons still aren't seen at native scale unless the whole 4000×3000 image fits, which is impractical on T4.
3. **"How do you justify Plan A as a contribution vs. just engineering?"** — The coupling between density-aware training and SAHI inference is the novel finding; naive SAHI without crop training fails catastrophically (0.255 mAP@0.5), proving the two halves are non-trivially interdependent.
4. **"Phase 3 negative result — won't reviewers reject the paper?"** — Plan A is the contribution; Phase 3 is reported in a lessons-learned subsection as evidence of a non-trivial methodological pitfall. Workshop venues value honest negative results.
5. **"What's the path to real-time deployment?"** — Plan A inference: ~3 seconds per 4000×3000 image with SAHI on a T4. Not real-time yet. Phase 4 post-processing modules are offline-only by design. Real-time engineering is left as future work after publication.
