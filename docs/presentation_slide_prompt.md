# Prompt to Generate Slides on Claude.ai

This document contains two prompts. The first builds a self-contained reveal.js HTML slide deck (recommended). The second builds a plain-text outline you can paste into PowerPoint / Google Slides.

Audience for the deck: 3 people with ML background but **no prior context on this project**. The deck should be **visually clean and pedagogical** — explain technical terms briefly on first use, use comparison tables where possible, avoid jargon walls. Every slide should answer three questions in this order: (1) *why are we looking at this*, (2) *what did we do*, (3) *what does the result mean*.

---

## PROMPT — reveal.js HTML slide deck (recommended)

Copy everything between the two `===` lines and paste into Claude.ai (works best with Opus 4.7 or Sonnet 4.6 + Artifacts enabled).

===

Create a 10-slide presentation deck for a 10-minute academic talk on a UAV
Search-and-Rescue person detection research project. Output it as a single
self-contained HTML file using reveal.js. The audience has ML background
but no prior context on this project — explanations should set context
clearly, every technical term gets a one-line plain-English definition on
first use, and the deck must be visually clean.

Style requirements:
- Use reveal.js from CDN (jsdelivr or unpkg). White background, dark text,
  blue accent (#2563eb), red accent for negative results (#dc2626).
- Sans-serif system font stack (-apple-system, "Segoe UI", Roboto).
- Monospace font for numerical metrics in tables.
- One screen at 1280×720, no scrolling.
- Use fragments (incremental reveal) only on slide 7 (Plan A results) and
  slide 8 (negative results table). Everything else appears at once.
- No emoji clutter. Allowed unicode: arrows (→, ↑, ↓), star (⭐) for the
  one contribution slide, X mark for negative results.
- Each slide title MUST be a complete short phrase ("Three difficulties",
  "The fix: scale alignment"), not a one-word label.
- Include a tiny page number at bottom-right of each slide (slide N / 10).

Content of each slide — use exactly this content, do not paraphrase
numerical values:

===
SLIDE 1 — Title
Title (large): "UAV Search & Rescue Person Detection"
Subtitle (medium): "Improving Aerial Small-Object Detection through a
                    Scale-Aligned Training Pipeline"
One-line headline (small italic): "YOLOv12-s + density-aware crops + SAHI:
mAP@0.5 0.759 → 0.872 on HERIDAL, without modifying the detector."
Footer: "Author / Date / Institution" placeholders.

===
SLIDE 2 — The scenario
Title: "The Scenario: A Lost Hiker"
Layout: simple narrative on the left, comparison on the right.
LEFT — three short paragraphs of running text (no bullets):
1. "A hiker doesn't return by sunset. The search radius around their
    last-known location grows quadratically with elapsed time — within
    hours, dozens of square kilometers must be covered."
2. "Ground teams cover 3-4 km/h. A drone covers 30 km/h — an order of
    magnitude faster."
3. "But a human operator cannot reliably spot a missing person in hours
    of live drone footage. Automated detection is the bottleneck."
RIGHT — a small "Search coverage per hour" table:
| Method | Coverage |
| Ground team (1 person) | ~3 km² |
| Drone + operator (manual review) | ~10 km² |
| Drone + automated detection | ~30 km² |
Bottom italic line: "This is the problem we want to solve."

===
SLIDE 3 — The three difficulties
Title: "Why It's Hard: Three Constraints That Shape Every Decision"
Three vertical cards (side by side, equal width):

CARD 1 (header in blue) — "1. Tiny objects"
"A person is ~30 px in a 4000×3000 image."
"That's <0.1% of image area."
"For a 6-inch phone screen, a 30-px person ≈ a fingertip nail."

CARD 2 (header in blue) — "2. Natural decoys"
"Rocks, huts, vegetation share person-like silhouettes from above."
"False positives in SAR are expensive — send teams to wrong locations."

CARD 3 (header in blue) — "3. Scarce data"
"HERIDAL: 1,124 training images."
"COCO: 118,000 training images."
"100× smaller — anything data-hungry is off the table."

Bottom line in bold: "Every design choice in this project follows from
these three constraints."

===
SLIDE 4 — The research framing
Title: "Two Possible Approaches"
Layout: two columns, side by side.

LEFT column (slightly faded):
Header: "Path A — Design a better detector"
- New architecture / attention modules
- New loss function
- Custom backbone
- Common in aerial small-object literature
- Risk: novel architecture overfits in small-data regime

RIGHT column (highlighted with blue border):
Header: "Path B — Improve everything AROUND a fixed detector ⭐"
- Same YOLOv12-s, never modified
- Train data composition
- Training preprocessing
- Inference strategy (SAHI)
- Post-processing
- "This is what we chose."

Bottom callout box:
"Why Path B?
(a) every change cleanly attributable to one component,
(b) pipeline portable to any detector,
(c) data-pipeline work generalizes better at small N."

===
SLIDE 5 — Two datasets in sequence
Title: "Two Datasets, Two Purposes"
Comparison table (clean, monospace numbers):
|                  | VisDrone (Phase 1)   | HERIDAL (Phase 2+) |
| Domain           | Urban aerial         | Wilderness aerial   |
| Typical scene    | Streets, vehicles    | Forest, rocks       |
| Train images     | 5,684                | 1,124               |
| Person size      | varied 30-80 px      | uniform ~30 px      |
| Role in project  | Methodology validation + model-size ablation | Actual SAR target |
| YOLOv12-s mAP@0.5| 0.552                | 0.759               |
Caption below table:
"VisDrone is the smoke test. HERIDAL is where every contribution after
Phase 1 is measured."
Sub-caption (italic, smaller):
"Note: HERIDAL outperforms VisDrone with 5× less data — wilderness
backgrounds are semantically simpler than urban ones."

===
SLIDE 6 — The hidden scale problem
Title: "The Hidden Problem: Train–Inference Scale Mismatch"
Three panels stacked vertically (or 3 boxes in one row, whichever fits):

PANEL 1 (gray box):
"What standard detectors do"
"4000×3000 image → resize → 640×640 → detector"
"30-px person becomes ~5 px after resize"
"Model never sees what persons look like at real scale."

PANEL 2 (gray box):
"The natural fix: SAHI (ICIP 2022)"
"At inference: cut original image into 640×640 TILES at native resolution"
"Persons appear at real 30-px size in each tile"
"Run detector on each tile, merge predictions"

PANEL 3 (red border, marked ❌):
"What happens when we apply SAHI to the Phase 2 model"
"mAP@0.5: 0.759 → 0.255"
"The model trained on 5-px persons sees 30-px persons at inference."
"Out-of-distribution input → catastrophic hallucination."
Bottom italic line: "SAHI alone is not enough. The model also has to be
TRAINED at the right scale."

===
SLIDE 7 — Plan A (the main contribution) ⭐
Title: "The Fix: Plan A — Couple Training and Inference Scales ⭐"

Top section — METHOD (3 numbered steps in a horizontal flow):
Step 1: "Crop training images"
"4000×3000 image → generate three 640×640 crops (2 person-centered + 1 random)"
"NO resizing — crops are at native pixel scale"
"4,500 crops from 1,124 images"

Step 2: "Train YOLOv12-s on crops"
"Same hyperparameters as baseline, 80 epochs"

Step 3: "Inference with SAHI"
"On original 4000×3000 val images, sliced into 640×640 tiles at native res"

Middle callout (highlighted blue):
"Now training and inference see the same scale. The two halves are coupled."

Bottom — RESULTS table (use fragments to reveal rows one by one):
| Metric | Phase 2 baseline | Naive SAHI ❌ | **Plan A ⭐** | Δ vs baseline |
| mAP@0.5 | 0.759 | 0.255 | **0.872** | **+0.113 (+15%)** |
| mAP@0.5:0.95 | 0.344 | 0.129 | **0.573** | **+0.229 (+66%)** |
| AP_small | — | 0.002 | **0.494** | ~250× over naive SAHI |

Final fragment (revealed last) — large quote box at bottom:
"Density-aware INFERENCE works only when paired with density-aware
TRAINING. Treating SAHI as a standalone trick is what made the first
attempt fail."

===
SLIDE 8 — Three negative results
Title: "Three Honest Negative Results — A Convergent Lesson"

Top intro line:
"After Plan A worked, three attempts to add more on top all underperformed.
Different mechanisms, different reasons, one common lesson."

Three rows (each row = one attempt). Use fragments to reveal one row at a
time. Each row has 4 columns:
                                 [Method]         [Result]        [Why it failed]

Row 1 — Phase 3: Synthetic data augmentation
| SDXL backgrounds + person | mAP@0.5: 0.013 ❌ | Mixed synthetic with raw 4000×3000 train images at imgsz=640 → reintroduced the same scale mismatch Plan A solved |

Row 2 — Phase 4 Module 1: SAM2 box refinement
| Prompt SAM2 with each Plan A bbox → refit tight box from mask | mAP@0.75: 0.661 → 0.492 ❌ | SAM2 segments tighter than HERIDAL GT by 2-3 px. At ~30-px person scale, that's enough to drop IoU below strict thresholds |

Row 3 — Phase 4 Module 2: Cascade classifier
| MobileNetV3-Small filter trained on Plan A TP vs FP | AP_small: 0.494 → 0.379 ❌ | 64×64 classifier input too small for confident classification at ~30-px person scale → drops small-scale TPs |

Bottom callout box (blue border):
"Convergent lesson: Plan A's scale alignment is the load-bearing piece.
Post-hoc additions that don't preserve scale-awareness break something —
each for its own specific mechanical reason."

===
SLIDE 9 — The full picture
Title: "Where Things Stand (YOLOv12-s held constant throughout)"
Visual: a horizontal bar chart of mAP@0.5 per phase, OR a clean table.

Table version (preferred for legibility):
| Phase | What was added | mAP@0.5 | Δ |
| 1. VisDrone baseline | — | 0.552 | — (reference) |
| 2. HERIDAL baseline | switched dataset | 0.759 | +0.207 |
| 2.5. Plan A ⭐ | density crops + SAHI | **0.872** | **+0.113** (main contribution) |
| 3. Synthetic data | SDXL + paste | 0.013 ❌ | -0.859 |
| 4. SAM2 refinement | post-hoc segment | 0.871 (-0.001) | mostly neutral on mAP@0.5, breaks strict thresholds |
| 4. Cascade classifier | post-hoc filter | 0.869 (-0.003) | breaks AP_small |

Bottom line (bold, blue):
"Plan A is the contribution. Negative results are documented, not
buried, because they collectively explain WHY scale-awareness is the
central design constraint."

===
SLIDE 10 — Takeaways
Title: "Three Takeaways"

Three numbered bullets in large text, with one-line explanation under each:

1. "Train–inference distribution alignment is the dominant lever in
    aerial small-object detection — more so than architectural change."
    (sub-line: "Plan A's gain comes from this alone, no new architecture.")

2. "SAHI is only effective when paired with native-resolution training."
    (sub-line: "The two halves are coupled. Naive SAHI alone collapses.")

3. "Foundation models do not transfer for free to small-aerial-person
    settings."
    (sub-line: "Both SDXL and SAM2 required scale-aware adaptation we
    haven't built yet.")

Bottom centered: "Thank you. Questions?"
Tiny footer line: "Code, weights, all results (positive and negative):
github.com/Lucas-on-the-cloud/UAV_res"

===

Hard requirements:
- All numerical values EXACTLY as written above. Do not round 0.872 to 0.87.
- One self-contained HTML file. CDN imports for reveal.js only — no extra
  assets, no inline images.
- Slide 7's results table and slide 8's negative results table both use
  reveal.js fragments to reveal rows progressively (the presenter wants to
  pace the reveal).
- No JavaScript animation beyond reveal.js's built-in slide transitions
  and fragments.
- Page number "N / 10" in bottom-right of each slide.

===

End of prompt.

---

## ALTERNATE PROMPT — outline only (PowerPoint / Google Slides)

If you'd rather build the deck yourself, use this prompt instead. Output
is plain markdown — copy each slide manually into your slide tool.

===

Generate a 10-slide outline for a 10-minute academic presentation on a
UAV Search-and-Rescue person detection research project. The audience
has ML background but no prior context. Every slide must answer (in
order): why we're looking at this, what was done, what the result means.

For each slide, provide:
1. Title (full short phrase, not a label)
2. 3-5 plain-English bullets, max 14 words each
3. Any tables or numerical data exactly as written
4. A 1-sentence speaker note on what to emphasize

Topics in order:
1. Title + headline (mAP 0.759 → 0.872 on HERIDAL, no detector change)
2. The SAR scenario (lost hiker, drones, human operators are the bottleneck)
3. Three difficulties: tiny persons, natural decoys, scarce data
4. Path A vs Path B research framing (we chose Path B — improve around fixed detector)
5. Two datasets: VisDrone for validation, HERIDAL as target
6. The hidden scale problem + why naive SAHI fails
7. Plan A — the main contribution (couple training and inference scales)
8. Three negative results: synthetic, SAM2, cascade — convergent lesson
9. Full results table across phases
10. Three takeaways

Hard requirement: use these exact numerical values throughout.
- Phase 1 VisDrone: 0.552
- Phase 2 HERIDAL: 0.759
- Plan A: mAP@0.5 = 0.872, mAP@0.5:0.95 = 0.573, AP_small = 0.494
- Naive SAHI failure: 0.255
- Phase 3 synthetic: 0.013
- Phase 4 SAM2: mAP@0.5 = 0.871, mAP@0.5:0.95 = 0.487, mAP@0.75 = 0.492
- Phase 4 Cascade: mAP@0.5 = 0.869, AP_small = 0.379

Output as plain markdown, one section per slide.

===

---

## After Claude generates the deck

**Reveal.js HTML version:**
1. Save Claude's output as `presentation.html`
2. Double-click to open in Chrome / Edge / Firefox
3. Press `F` for full-screen
4. `→` / `←` to navigate slides; `↓` / `↑` for fragments inside a slide; `Esc` for slide overview; `S` for speaker notes view

**Outline version:**
1. Copy each slide's content into PowerPoint / Google Slides manually
2. Use a clean academic theme (PowerPoint "Banded" or Google Slides "Material Light")
3. ~15 minutes of formatting

---

## Delivery tips

- **Slide 7** is the main contribution slide. Stay on it for ~75 seconds. The audience will read the numbers — give them time.
- **Slide 8** is the longest at ~80 seconds. It's also the slide where committee members will probe in Q&A. Walk through each negative result calmly — don't apologize for them.
- **Don't read the slides verbatim.** Use the [presentation_script_10min.md](presentation_script_10min.md) for the spoken version. Slides are backdrop.
- **Practice once with a timer.** Target 9:15 spoken, leaves ~45 seconds buffer.
- **For Q&A**, the script's bottom section has 7 pre-prepared one-sentence answers to the most likely committee challenges. Skim those before walking in.
