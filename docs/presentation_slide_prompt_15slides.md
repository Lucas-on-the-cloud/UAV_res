# Slide Prompt — 15 slides, 3 presenters

Paste everything between the `===` markers into claude.ai (Opus 4.7 or Sonnet 4.6 with Artifacts). Output is a single self-contained HTML file using reveal.js, ~10 minute talk split across 3 presenters.

===

Create a 15-slide reveal.js HTML presentation for a 10-minute academic talk
on a UAV Search-and-Rescue person detection research project. Three presenters
share the deck — slides 1-5 (Presenter A), slides 6-10 (Presenter B),
slides 11-15 (Presenter C). The audience has ML background but no prior
context on this project.

STYLE RULES (strict):
- Single self-contained HTML file, reveal.js + clean white theme from CDN.
- Sans-serif system font, blue accent (#2563eb), red accent for negative
  results (#dc2626), green accent (#16a34a) for the main contribution.
- Each slide title is a complete short phrase, NOT a label.
- Slide text is DIRECT — what's on the slide is exactly what gets said.
  No padding, no speaker notes section, no "see notes for details".
- Show the presenter handoff with a small badge in top-right corner of
  each slide: "Presenter A" (slides 1-5, blue), "Presenter B" (6-10,
  green), "Presenter C" (11-15, red).
- Page number "N / 15" in bottom-right.
- Tables and metric numbers in monospace font.
- One screen at 1280×720, no scrolling. Keep text tight.
- Use reveal.js fragments ONLY for slide 10 (Plan A results table) and
  slide 13 (three negative results comparison).

CONTENT — use these EXACT numbers, do not paraphrase.

===
SLIDE 1 (Presenter A) — Title
Title: "UAV Search & Rescue Person Detection"
Subtitle: "A scale-aligned pipeline for aerial small-object detection"
Headline result (small italic box): "YOLOv12-s, unchanged. Pipeline lifts
HERIDAL mAP@0.5 from 0.759 to 0.872."
Footer: "Presenter A / B / C — Date — Institution"

===
SLIDE 2 (Presenter A) — The problem
Title: "The SAR Detection Problem"
Three lines of running text:
"A hiker goes missing. Search radius grows quadratically with time."
"Ground teams cover 3-4 km/h. A drone covers 30 km/h — 10× faster."
"But human operators cannot reliably spot tiny persons in hours of live
footage. Automated detection is the real bottleneck."
Bottom callout (bold blue):
"Task: detect every person in a 4000×3000 aerial RGB image, with tight
bounding boxes, in real time."

===
SLIDE 3 (Presenter A) — Why it's hard
Title: "Three Constraints"
Three horizontal cards (equal width):

CARD 1 — "Tiny objects"
"Person ≈ 30 px in 4000×3000 image"
"< 0.1% of image area"
"On a phone screen, like a fingertip nail"

CARD 2 — "Natural decoys"
"Rocks, huts, vegetation look like persons from above"
"False positives = wasted SAR resources"

CARD 3 — "Scarce data"
"HERIDAL: 1,124 train images"
"COCO: 118,000 train images"
"100× smaller — data-hungry methods are out"

Bottom line (bold): "These three constraints drive every design choice."

===
SLIDE 4 (Presenter A) — Our approach
Title: "Our Approach: Pipeline, Not New Architecture"
Two columns side by side:

LEFT (faded gray): "Common path — design a new detector"
- New architecture / attention modules
- New loss
- Risk: overfits in 1,124-image regime

RIGHT (highlighted blue border): "Our path — improve AROUND a fixed detector"
- YOLOv12-s, released weights, never modified
- Change training data, preprocessing, inference, post-processing
- Every Δ attributable to one component
- Pipeline portable to any detector

===
SLIDE 5 (Presenter A) — Two datasets, two roles
Title: "Two Datasets in Sequence"
Comparison table:
|                  | VisDrone (Phase 1)    | HERIDAL (Phase 2+)  |
| Domain           | Urban aerial          | Wilderness aerial   |
| Train images     | 5,684                 | 1,124               |
| Person size      | varied                | uniform ~30 px      |
| Role             | Methodology validation| Actual SAR target   |
|                  | + model size ablation | All improvements    |
|                  |                       | measured here       |

Bottom line: "VisDrone confirms the pipeline works. HERIDAL is where we
measure progress."

[HANDOFF TO PRESENTER B]

===
SLIDE 6 (Presenter B) — Phase 1: pick the model size
Title: "Phase 1 — VisDrone: YOLOv12-n vs YOLOv12-s"
Result table:
| Metric       | YOLOv12-n | YOLOv12-s |
| mAP@0.5      | 0.476     | **0.552** |
| mAP@0.5:0.95 | 0.187     | **0.231** |
| Recall       | 0.438     | **0.497** |

Plain text below:
"YOLOv12-s wins by +0.076 mAP@0.5 at modest compute cost."
"Locked in for every subsequent phase. No more model-size changes."

===
SLIDE 7 (Presenter B) — Phase 2: HERIDAL baseline
Title: "Phase 2 — YOLOv12-s on HERIDAL"
Result table:
| Metric         | VisDrone (Phase 1) | HERIDAL (Phase 2) |
| mAP@0.5        | 0.552              | **0.759**         |
| mAP@0.5:0.95   | 0.231              | **0.344**         |
| Recall         | 0.497              | **0.713**         |

Bottom text (split into two parts):
"HERIDAL is easier than VisDrone — wilderness backgrounds are simpler."
"BUT mAP@0.5:0.95 = 0.344 is low: model FINDS persons (mAP@0.5 = 0.759)
but does NOT localize them tightly. Why?"

===
SLIDE 8 (Presenter B) — The hidden problem
Title: "The Hidden Problem: Train–Inference Scale Mismatch"

Top panel (gray box):
"What standard training does:"
"4000×3000 image → resize to 640×640 → train detector"
"30-px person becomes ~5 px in training input"
"Model never sees persons at their real scale"

Middle panel (gray box):
"The natural fix: SAHI sliced inference"
"Cut 4000×3000 into 640×640 tiles AT NATIVE RESOLUTION"
"Persons appear at their real 30-px size in each tile"

Bottom panel (red border, large X mark):
"Applying SAHI to Phase 2 model:"
"mAP@0.5: 0.759 → 0.255  ❌"
"The model trained on 5-px persons cannot recognize 30-px persons."
"Out-of-distribution input → catastrophic failure."

===
SLIDE 9 (Presenter B) — Plan A method
Title: "Plan A — Couple Training and Inference Scales ⭐"

Three numbered steps in a horizontal flow:

STEP 1: "Generate 640×640 crops"
"From each 4000×3000 train image:
2 person-centered + 1 random crop, native resolution, no resize"
"4,500 crops from 1,124 images"

STEP 2: "Train YOLOv12-s on crops"
"Same hyperparameters as baseline, 80 epochs, AdamW"

STEP 3: "SAHI inference on val"
"Original 4000×3000 images → 640×640 tiles, native resolution"

Middle callout (highlighted green):
"Training distribution = Inference distribution.
Model sees what persons actually look like."

===
SLIDE 10 (Presenter B) — Plan A results ⭐
Title: "Plan A Results — Main Contribution"

Results table (use fragments to reveal rows one by one):
| Metric          | Phase 2 baseline | Naive SAHI ❌ | **Plan A ⭐**  | Δ vs baseline       |
| mAP@0.5         | 0.759            | 0.255         | **0.872**     | **+0.113 (+15%)**   |
| mAP@0.5:0.95    | 0.344            | 0.129         | **0.573**     | **+0.229 (+66%)**   |
| mAP@0.75        | —                | 0.117         | **0.661**     | (large)             |
| AP_small        | —                | 0.002         | **0.494**     | ~250× over naive SAHI |
| AR_100          | —                | 0.438         | **0.662**     | +0.224              |

Quote box at bottom (revealed last):
"Density-aware INFERENCE works only when paired with density-aware
TRAINING. The two halves are coupled."

[HANDOFF TO PRESENTER C]

===
SLIDE 11 (Presenter C) — Trying to push further
Title: "Phase 3 + Phase 4: Adding More on Top of Plan A"

Plain text:
"Plan A established a strong baseline (mAP@0.5 = 0.872)."
"We tried three more enhancements. All underperformed."
"Each fails for a DIFFERENT mechanical reason. The pattern is the
interesting finding."

Preview line (small):
"Next 3 slides: synthetic data → SAM2 refinement → cascade classifier."

===
SLIDE 12 (Presenter C) — Phase 3: Synthetic data augmentation
Title: "Phase 3 — Synthetic Data (Failed)"

Method box:
"Generate 2,000 aerial backgrounds with Stable Diffusion XL"
"Composite real HERIDAL persons onto them"
"Mix into training together with raw HERIDAL images"

Result table:
| Method                   | mAP@0.5 |
| Plan A baseline          | 0.872   |
| Phase 3 SDXL synthetic   | **0.013 ❌** |

Diagnosis box (italic):
"Mixing synthetic data with raw 4000×3000 HERIDAL at imgsz=640
RE-INTRODUCED the same scale mismatch Plan A solved."
"The synthetic data was fine. The pipeline consuming it was wrong."

===
SLIDE 13 (Presenter C) — Phase 4: Two post-processing modules
Title: "Phase 4 — SAM2 and Cascade Classifier (Both Negative)"

Side-by-side comparison (use fragments to reveal each side):

LEFT BOX — "Module 1: SAM2 bbox refinement"
Method: "Box-prompt SAM2 with each Plan A bbox → segment person → refit
tight bbox from mask"
Result:
| Metric        | Plan A | + SAM2 |
| mAP@0.5       | 0.872  | 0.871 |
| mAP@0.5:0.95  | 0.573  | 0.487 |
| mAP@0.75      | 0.661  | **0.492 ❌** |
Diagnosis (small): "SAM2 refit boxes are 2-3 px tighter than HERIDAL GT.
At 30-px persons, IoU drops below strict thresholds."

RIGHT BOX — "Module 2: Cascade classifier"
Method: "MobileNetV3-Small trained on Plan A's TP/FP crops, filters
predictions at inference"
Result:
| Metric        | Plan A | + Cascade |
| mAP@0.5       | 0.872  | 0.869 |
| AP_small      | 0.494  | **0.379 ❌** |
| Filter rate   | —      | 38.9% |
Diagnosis (small): "64×64 input too small for confident classification
at 30-px person scale. Cascade drops small-scale TPs."

===
SLIDE 14 (Presenter C) — The convergent lesson
Title: "Three Failures, One Lesson"

Three rows in a table:
| Attempt                    | Why it failed                                |
| Phase 3 synthetic data     | Scale mismatch in training                   |
| Phase 4 SAM2 refinement    | Box-tightness mismatch with GT convention    |
| Phase 4 cascade classifier | Signal-to-noise limit at small person scale  |

Bottom callout (large, bold, green border):
"Plan A's scale alignment is the LOAD-BEARING piece.
Any post-hoc addition that doesn't preserve scale-awareness breaks
something — each for a different specific reason."

===
SLIDE 15 (Presenter C) — Takeaways
Title: "Three Takeaways"

Three numbered points (large text):

1. "Train-inference distribution ALIGNMENT is the dominant lever in
    aerial small-object detection — more than architectural change."
    (sub-line: "Plan A's +0.113 mAP comes from this alone, no new model.")

2. "SAHI works ONLY when paired with native-resolution training."
    (sub-line: "Naive SAHI alone: 0.759 → 0.255. The halves are coupled.")

3. "Foundation models (SDXL, SAM2) don't transfer for free here."
    (sub-line: "Both required scale-aware adaptation we haven't built yet.")

Bottom centered:
"Thank you. Questions?"
Tiny footer: "Full code, weights, positive + negative results:
github.com/Lucas-on-the-cloud/UAV_res"

===

HARD REQUIREMENTS:
- All numbers EXACTLY as written above. No rounding (0.872 stays 0.872).
- Self-contained HTML file. CDN imports only — no inline images, no
  external assets.
- Presenter badge in top-right of every slide.
- Page number "N / 15" in bottom-right of every slide.
- Slides 10 and 13 use fragments; everything else appears at once.
- No JS animation beyond reveal.js defaults.

===
