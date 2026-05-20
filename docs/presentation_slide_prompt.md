# Prompt to Generate Slides on Claude.ai

Copy the block below into a new Claude.ai conversation (any model, but Claude Opus 4.7 or Sonnet 4.6 with Artifacts gives the cleanest result). Claude will produce a single-file HTML reveal.js slide deck you can open in a browser and present full-screen.

If you'd rather have a Google Slides / PowerPoint structure (text content only, you import into a deck manually), use the **alternate prompt** at the bottom.

---

## PROMPT — reveal.js HTML slide deck (recommended)

```
Create a 10-slide presentation deck for a 10-minute academic talk on a
UAV Search-and-Rescue person detection research project. Output it as a
single self-contained HTML file using reveal.js (load reveal.js + a clean
white theme from CDN — jsdelivr or unpkg). The deck should be ready to
open in a browser and present in full-screen.

Style requirements:
- Clean, academic style. White background, dark text, one accent color
  (use #2563eb blue for highlights).
- Sans-serif (system font stack: -apple-system, "Segoe UI", Roboto).
- Tables and metric numbers must be clearly readable; use monospace
  font for numbers in tables.
- Use simple unicode arrows (→) and the star symbol (⭐) for the
  contribution highlight. NO emoji clutter.
- Each slide should fit on one screen at 1280x720 without scrolling.
- Use reveal.js fragments (incremental reveal) sparingly — only for the
  results bullets on slide 6.

Content of each slide (use exactly this content, do not paraphrase the
numbers):

---
SLIDE 1 — Title
Title: "UAV Search & Rescue Person Detection"
Subtitle: "Density-aware Crops + SAHI on YOLOv12-s"
Footer line: "Author / Date / Institution" (leave as placeholders)

---
SLIDE 2 — The Problem
Header: "The Problem"
Three bullets:
- "Tiny persons: ~30 px in 4000×3000 aerial image (<0.1% area)"
- "Natural distractors: rocks, huts, vegetation look like persons"
- "Scarce labels: HERIDAL = 1,124 train images (100× smaller than COCO)"
Bottom italic line: "Why this matters: search radius grows quadratically
with time. Drones sweep faster than ground teams — but only if detection
is automated."

---
SLIDE 3 — Approach
Header: "Approach: a Pipeline, Not a New Detector"
One key sentence in large text: "YOLOv12-s is held CONSTANT across every
phase. The contribution is outside-the-model."
Below, three-layer list:
1. Layer 1 — Dataset selection (Phase 1, 2)
2. Layer 2 — Train–inference scale alignment ⭐ (Phase 2.5, main contribution)
3. Layer 3 — Post-processing (Phase 4, in progress)

---
SLIDE 4 — Layer 1: Datasets
Header: "Layer 1 — Why Two Datasets?"
Two-column table:
| | VisDrone (Phase 1) | HERIDAL (Phase 2+) |
| Domain | Urban aerial | Wilderness aerial |
| Role | Methodology validation + n vs s ablation | Actual SAR target |
| mAP@0.5 (YOLOv12-s) | 0.552 | 0.759 |
Caption: "HERIDAL outperforms VisDrone with 5× less data — wilderness
backgrounds are semantically simpler than urban."

---
SLIDE 5 — The Localization Problem
Header: "Phase 2 Baseline: Found, But Not Tight"
Bullet 1: "mAP@0.5 = 0.759 (finds persons) BUT mAP@0.5:0.95 = 0.344 (loose boxes)"
Bullet 2: "Cause: imgsz=640 on 4000×3000 image → 30-px person shrinks to ~5 px"
Bullet 3: "Naive SAHI fix (slice tiles at native res): mAP@0.5 0.759 → 0.255 ❌"
Bullet 4: "Diagnosis: training–inference resolution mismatch"

---
SLIDE 6 — Plan A (the main contribution) ⭐
Header: "Layer 2 — Plan A: Density Crops + SAHI ⭐"
Method box (3 steps, numbered):
1. Generate 640×640 native-resolution crops from train (2 person-centered + 1 random per image)
2. Train YOLOv12-s on crops, 80 epochs
3. SAHI sliced inference on 4000×3000 val (640 tiles, 0.2 overlap)
Results table (use fragments to reveal one row at a time):
| Metric | Phase 2 baseline | Naive SAHI | Plan A ⭐ | Δ |
| mAP@0.5 | 0.759 | 0.255 | 0.872 | +0.113 |
| mAP@0.5:0.95 | 0.344 | 0.129 | 0.574 | +0.229 |
| AP_small | — | 0.002 | 0.494 | ~250× |
Final fragment as a quote box at bottom:
"Key insight: density-aware INFERENCE only works when the model has
seen density-aware TRAINING samples. The two must be coupled."

---
SLIDE 7 — Phase 3: A Negative Result (honest)
Header: "Phase 3 — Synthetic Augmentation (negative result)"
Two-column table:
| Approach | mAP@0.5 |
| A. SDXL synthetic + raw HERIDAL train | 0.013 |
| B. HERIDAL composite + raw HERIDAL train | 0.000 |
| Plan A baseline (unchanged) | 0.872 |
Diagnosis box:
"Both approaches mixed synthetic data with RAW 4000×3000 HERIDAL
training images at imgsz=640. This reintroduced the same scale
mismatch Plan A solved. Synthetic samples were fine — the pipeline
consuming them was wrong."
Bottom line in italics:
"Future work: 'crop-then-synthesize' — apply Plan A's crop pipeline
to synthetic samples too."

---
SLIDE 8 — Phase 4 (both modules tested, both underperformed)
Header: "Layer 3 — Phase 4: Two post-processing modules, both negative"
Combined results table:
| Metric | Plan A | + SAM2 | + Cascade |
| mAP@0.5 | 0.872 | 0.871 (-0.001) | 0.869 (-0.003) |
| mAP@0.5:0.95 | 0.573 | 0.487 (-0.086) | 0.571 (-0.002) |
| mAP@0.75 | 0.661 | 0.492 (-0.169) | 0.663 (+0.002) |
| AP_small | 0.494 | 0.450 (-0.044) | 0.379 (-0.115) |
Diagnosis box (small italic text, two lines):
"SAM2: refit boxes 2-3 px tighter than HERIDAL GT - IoU drops below
strict thresholds at ~30 px person scale."
"Cascade: filters 39% of preds, but disproportionately drops small-scale
TPs - 64x64 input too small for confident classification."
Bottom line in bold:
"Common lesson: Plan A is scale-tuned for HERIDAL. Post-hoc additions
without explicit scale-awareness break the scale alignment."

---
SLIDE 9 — Summary Contribution Stack
Header: "Cumulative Contribution Stack (YOLOv12-s held constant)"
Vertical bar chart OR table showing mAP@0.5 progression:
- Phase 1 (VisDrone): 0.552
- Phase 2 (HERIDAL): 0.759 (Δ +0.207, dataset switch)
- Phase 2.5 (Plan A): 0.872 (Δ +0.113, training+inference pipeline) ⭐
- Phase 3 (synth aug): failed (scale mismatch)
- Phase 4 SAM2: 0.871 (over-tightens bbox)
- Phase 4 Cascade: 0.869 (drops small-scale TPs)
Bottom line: "Plan A is the contribution (mAP@0.5 = 0.872). Three
documented negative results converge on one lesson: in this regime,
scale-awareness is the load-bearing constraint."

---
SLIDE 10 — Closing
Header: "Takeaways"
Three numbered bullets:
1. "In aerial small-object detection, train–inference distribution
   alignment is the dominant lever — more so than architectural change."
2. "SAHI is only effective when paired with native-resolution training
   data. The two are coupled, not independent."
3. "Drop-in foundation models (SDXL, SAM2) do NOT transfer for free
   to this domain. Both Phase 3 and Phase 4 SAM2 required scale-aware
   adaptation that wasn't free."
Bottom: "Thank you. Questions?"
Tiny footer line in light grey: "Code, weights, both positive and
negative results: github.com/Lucas-on-the-cloud/UAV_res"

---

Make sure:
- All numbers are EXACTLY as given above (don't round 0.872 to 0.87, etc.).
- The HTML file is fully self-contained — no separate assets — just CDN
  imports for reveal.js. I should be able to save it as one .html file
  and open it.
- Slide 6 uses incremental reveal (data-fragment-index) for the result
  rows; everything else appears at once.
- No JavaScript animation or images — just text, tables, and reveal.js's
  built-in slide transitions.
```

---

## ALTERNATE PROMPT — outline only (for PowerPoint / Google Slides import)

Use this if you prefer building the deck yourself in PowerPoint/Slides and just want structured content per slide:

```
Generate a 10-slide outline for a 10-minute academic presentation on a
UAV Search-and-Rescue person detection research project. For each slide,
give me:

1. Slide title (max 10 words)
2. 3-5 concise bullets (max 12 words each)
3. Any tables or numerical data, written in plain text format
4. One "speaker note" line (1 sentence) describing what to emphasize

Topics in order:
1. Title slide
2. The problem (tiny persons in aerial SAR + scarce data)
3. Approach: pipeline, not new detector (YOLOv12-s held constant)
4. Why two datasets (VisDrone → HERIDAL)
5. Phase 2 baseline + naive SAHI failure
6. Plan A (Phase 2.5) — the main contribution
7. Phase 3 negative result on synthetic augmentation
8. Phase 4 planned work (SAM2 + cascade classifier)
9. Contribution stack summary
10. Closing takeaways

Key numbers to include (use exactly):
- Phase 1 VisDrone: mAP@0.5 = 0.552
- Phase 2 HERIDAL: 0.759
- Plan A: 0.872 (Δ +0.113 vs Phase 2; +66.5% relative on mAP@0.5:0.95)
- Plan A AP_small: 0.494 (≈250× vs naive SAHI)
- Phase 3 SDXL: 0.013, Phase 3 HERIDAL composite: 0.000

Output as plain markdown with one section per slide.
```

---

## What to do after Claude generates the deck

**For the reveal.js HTML version:**
1. Save Claude's output as `presentation.html`
2. Double-click to open in Chrome / Edge / Firefox
3. Press `F` for full-screen
4. Use arrow keys to navigate; `Esc` for overview; `S` for speaker notes

**For the outline version:**
1. Copy each slide's content into PowerPoint / Google Slides manually
2. Pick one academic theme (Microsoft has "Banded" or Google Slides has "Material Light")
3. ~15 minutes of formatting work

---

## Tips for the actual delivery

- **Slide 6 is the main slide** — stay on it for ~75 seconds. The committee will read the numbers.
- **Slide 7** — when discussing the negative result, lean in and slow down. Honest reporting of failure is a strength.
- **Don't read the slides verbatim.** Use the [presentation_script_10min.md](presentation_script_10min.md) for the spoken version and the slides as backdrop.
- **Practice once with a timer.** Aim for 8:30 spoken, leaving 1:30 buffer.
- **Anticipate Q&A** — the script doc has 5 pre-prepared answers at the bottom.
