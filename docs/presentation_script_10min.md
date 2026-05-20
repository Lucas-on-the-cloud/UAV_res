# 10-Minute Presentation Script — UAV SAR Person Detection

**Audience:** 3 people with ML background but no prior context on this project.
**Length target:** 10 minutes, ~1800 spoken words.
**Tone:** conversational, story-led, explain every technical term in plain English on first use.

**Pedagogical principle:** every slide answers three questions in order:
1. *Why are we looking at this?* (motivation, one sentence)
2. *What did we do?* (method, plain language)
3. *What does the result mean?* (interpretation, not just a number)

---

## [SLIDE 1 — Title]

*(Speak ~25 seconds)*

Good morning. I'm presenting a research project on **automated person detection in drone footage for Search and Rescue** — the situation where someone goes missing in the wilderness and a search team needs to find them.

The headline result is this: I built a pipeline on top of a standard off-the-shelf object detector. Without changing the detector itself, the pipeline lifts detection accuracy on the standard wilderness SAR benchmark from **76% to 87% mean Average Precision** — about a 15% relative improvement. I'll also show three honest negative results, because they're informative.

The talk is ten minutes. Please save questions for the end.

---

## [SLIDE 2 — Why we're working on this]

*(Speak ~55 seconds)*

Imagine a hiker who didn't come back to their trailhead by sunset. The search radius around their last-known location grows **quadratically** with elapsed time — so within a few hours, ground teams need to cover dozens of square kilometers. Ground teams walk at maybe three or four kilometers an hour. A drone flying at thirty kilometers an hour can scan that same area an order of magnitude faster.

But there's a catch: a human operator watching the drone's live video feed cannot reliably spot a missing person. The person is tiny in the frame, the operator gets fatigued, and a single missed detection in hours of footage costs lives. So the real bottleneck is not the drone — it's **automating** the visual detection.

The technical problem reduces to: given an aerial RGB image — roughly 4000 by 3000 pixels, taken from a drone flying maybe 50 meters above ground — find every person in the scene with a tight bounding box, in real time.

That sounds like ordinary object detection. Why is it actually hard? Three reasons.

---

## [SLIDE 3 — The three difficulties]

*(Speak ~55 seconds)*

**First — persons are tiny.** A typical person in this kind of aerial image is about **30 pixels wide**. The image is 4000 by 3000. So the person occupies less than 0.1% of the image area. To give you a physical sense: if the image were a phone screen, a 30-pixel person would be about the size of a single fingertip nail. Detectors trained on natural images, where objects fill a large fraction of the frame, struggle at this scale.

**Second — the wilderness is full of decoys.** Rocks, small huts and ruins, and dense vegetation patches share person-like silhouettes when viewed from above. A model that gets person-detection accuracy up too aggressively also starts firing on rocks. False positives in a SAR setting are expensive — they send searchers to wrong locations.

**Third — labelled data is scarce.** The standard wilderness SAR dataset, called **HERIDAL**, contains 1,124 training images. By comparison, the standard object-detection benchmark COCO contains 118,000 images. So we're working in a small-data regime. Anything that requires lots of labelled SAR data is off the table from day one.

Every design choice in the project is driven by these three constraints.

---

## [SLIDE 4 — The research framing]

*(Speak ~55 seconds)*

Now, a key methodological choice. There are two ways to attack a detection problem like this.

**Path A** is to design a better detector — invent a new architecture, add attention modules, new loss functions, custom backbone. This is the most common approach in the literature, especially in aerial small-object papers.

**Path B** is to take an existing strong detector as-is and improve everything **around** it — the training data, how it's preprocessed, how inference is run, how outputs are post-processed.

I chose Path B. The detector is **YOLOv12-small**, a recent real-time object detector from NeurIPS 2025. I use the released weights directly. I don't change the architecture, the loss, or the optimizer — **at any point in the project**.

Why? Three reasons. First, holding the detector fixed makes every result attributable to one specific change outside the model. Second, the pipeline is portable — anyone can swap in YOLOv11 or RT-DETR and see if it still helps. Third, in a small-data regime like ours, novel architectures tend to overfit; well-tuned data pipelines tend to generalize. Path B is the safer research bet.

---

## [SLIDE 5 — The two datasets: VisDrone and HERIDAL]

*(Speak ~55 seconds)*

A quick note on the data, because it matters for the story.

I train on two aerial datasets in sequence. The first is **VisDrone** — drone-shot images of urban streets, with cars, buildings, and crowds. It's a well-known benchmark. The second is **HERIDAL** — drone-shot images of wilderness areas, with forests, rocks, and isolated persons. HERIDAL is the actual SAR scenario.

So why train on both, if VisDrone is not the target?

VisDrone serves as **methodology validation**. It's well-benchmarked, it has plenty of data, and I can confirm my whole training-and-evaluation pipeline works correctly there before betting on the harder HERIDAL set. It's also where I picked YOLOv12-small over YOLOv12-nano — a model-size ablation that VisDrone has enough data to do reliably, but HERIDAL doesn't.

Once the pipeline is validated, I switch to HERIDAL — and from there, all reported improvements are measured on HERIDAL, because that's the actual SAR problem.

The Phase 2 result: vanilla YOLOv12-small fine-tuned on HERIDAL hits **mean Average Precision at 0.5 IoU of 0.759** — meaning at a moderately strict definition of "correct detection", the model is right 76% of the time. That's a reasonable starting point. But there's a hidden problem.

---

## [SLIDE 6 — The hidden problem: scale mismatch]

*(Speak ~60 seconds)*

Standard object detectors don't process 4000-by-3000 images directly — that's too large and too expensive. They first **resize** the image down to a smaller fixed size — typically 640 by 640 pixels. That means the 30-pixel person in the original gets shrunk down to about **5 pixels** in the training input. The detector never actually sees what a person looks like at the scale at which persons appear in real drone footage.

There's a published technique called **SAHI**, from ICIP 2022. SAHI says: at inference time, instead of resizing the big image, cut it into 640-by-640 **tiles** at native resolution and run the detector on each tile separately. That way persons appear at their actual 30-pixel size in each tile. Seems like a sensible fix.

I tried this. **It failed catastrophically** — mean Average Precision dropped from 0.759 to 0.255. The model started hallucinating hundreds of detections per tile.

The reason: the model was **trained** on tiny, 5-pixel persons. But SAHI fed it **native-resolution** patches at inference. The input was completely outside the model's training distribution. That's the diagnosis — a train-inference distribution mismatch. SAHI as a standalone fix is not enough.

The fix to this is the main contribution of the project.

---

## [SLIDE 7 — The fix: Plan A (training-and-inference coupled)]

*(Speak ~75 seconds — this is the main contribution slide, slow down)*

The insight is simple in hindsight. If you want SAHI to work at inference, you have to make sure the model sees the same kind of input — native-resolution 640-by-640 patches — **during training**, not just at inference.

So I changed the training data. For each 4000-by-3000 training image, I generate three crops of size 640 by 640 — two centered on actual person locations with random jitter, and one random crop. These crops are kept at native resolution; no resizing. I get about 4,500 training crops out of 1,124 original images. I fine-tune YOLOv12-small on those crops, same hyperparameters, 80 epochs. Then at inference, I run SAHI sliced inference on the original 4000-by-3000 validation images.

Now the **training distribution** — native-resolution 640 crops — matches the **inference distribution** — native-resolution 640 SAHI tiles. The detector has been shown, during training, exactly what it will see at deployment.

The result: mean Average Precision at 0.5 goes from 0.759 to **0.872** — a 15% relative improvement. mean Average Precision at the strict 0.5-to-0.95 setting goes from 0.344 to **0.573**, which is a 66% relative improvement in localization quality. And the metric I care about most — Average Precision on small objects — goes from essentially zero under naive SAHI to **0.494**. That's a 250-fold improvement on the small-person metric.

The lesson, which generalizes beyond this project: **density-aware inference works only when paired with density-aware training**. These two halves are coupled. Treating SAHI as a standalone post-hoc trick is what made my first attempt fail.

This is Plan A. It is the **main contribution** of the project.

---

## [SLIDE 8 — Three negative results worth reporting]

*(Speak ~80 seconds — this is long but important, lean in)*

After Plan A worked, I spent time trying to add more on top of it. Three attempts. All of them underperformed. I want to walk through them quickly because the pattern across them is the actual interesting finding.

**First attempt — synthetic data augmentation.** Generate two thousand additional aerial backgrounds with Stable Diffusion, paste real HERIDAL persons onto them, mix into training. The idea: more data, more diverse backgrounds, model generalizes better. **Result: mAP collapsed to essentially zero.** Why? I mixed the synthetic data with the raw 4000-by-3000 HERIDAL images at training time. That re-introduced the same scale mismatch Plan A had just solved.

**Second attempt — SAM2 bounding-box refinement.** SAM2 is Meta's recent foundation model for segmentation. I prompted SAM2 with each of Plan A's bounding boxes, asked it to segment the person, then refit a tight bounding box from the segmentation mask. The idea: tighter boxes, higher mAP at strict thresholds. **Result: mAP at strict thresholds dropped substantially.** Why? SAM2's segmentations are 2 to 3 pixels tighter than the HERIDAL ground-truth annotation convention. For 30-pixel persons, 2-3 pixels is a lot — it pushes IoU below the strict threshold.

**Third attempt — a cascade classifier.** Train a small MobileNet to distinguish person from rock-or-bush in Plan A's predictions, apply it as a confidence filter at inference. The idea: filter out the false positives we know Plan A produces. **Result: overall mAP unchanged, but small-object AP dropped from 0.494 to 0.379.** Why? At 64-by-64 classifier input, a 30-pixel person has too little signal for confident classification, and the classifier ends up filtering out small-scale true positives.

So three different mechanisms — synthetic data, segmentation refinement, learned filter — all underperform, each for a **different specific reason**: scale mismatch in training, box-tightness mismatch in refinement, signal-to-noise limits in small-scale classification. The convergent lesson is that **Plan A's scale alignment is the load-bearing piece**, and post-hoc additions that don't preserve scale-awareness break something.

---

## [SLIDE 9 — Where things stand]

*(Speak ~40 seconds)*

To summarize visually: the same YOLOv12-small detector, held fixed across every phase, gets these numbers on the HERIDAL test set.

Phase 1 on VisDrone gives 0.552 — that's the urban-aerial reference. Phase 2, dropping it on HERIDAL unchanged, gives 0.759 — wilderness is actually easier than urban for this detector because backgrounds are less cluttered. Phase 2.5 — Plan A, the density-aware crops plus SAHI — gives **0.872**. That's the headline.

Phase 3 synthetic augmentation, Phase 4 SAM2, Phase 4 cascade — all three underperformed Plan A by small or moderate margins, each for a documented mechanical reason.

Plan A is the contribution. The negative results are documented, not buried, because they collectively explain why scale-awareness is the central design constraint in this problem.

---

## [SLIDE 10 — Takeaways and questions]

*(Speak ~35 seconds)*

Three takeaways I'd like to leave with you.

**First** — in aerial small-object detection, the dominant lever isn't a new detector architecture. It's getting training and inference distributions to match. That's where the big numbers come from.

**Second** — SAHI-style sliced inference is only effective when paired with native-resolution training data. The two are coupled, not independent.

**Third** — drop-in foundation models — diffusion-generated data, SAM2 segmentation — do not transfer for free to small-aerial-person settings. Each requires explicit scale-aware adaptation.

All code, weights, both positive and negative results, are on the project GitHub repository. Thank you. Happy to take questions.

---

## Timing breakdown

| Slide | Topic | Time |
|---|---|---|
| 1 | Title + setup | 0:25 |
| 2 | Why SAR (the story) | 0:55 |
| 3 | Three difficulties | 0:55 |
| 4 | Path A vs Path B (research framing) | 0:55 |
| 5 | The two datasets (VisDrone → HERIDAL) | 0:55 |
| 6 | Scale mismatch + naive SAHI failure | 1:00 |
| 7 | **Plan A — main contribution** | 1:15 |
| 8 | Three negative results | 1:20 |
| 9 | Summary contribution stack | 0:40 |
| 10 | Takeaways | 0:35 |
| **Total** | | **~9:15** |

~45 seconds of buffer for transitions, breathing, asides. Don't memorize verbatim — internalize each slide's *one main point* and improvise. Slow down on slides 6, 7, and 8.

## Anticipated Q&A — short, internalized answers

1. **"Why YOLOv12 specifically? Why not RT-DETR or YOLOv11?"** — NeurIPS 2025, real-time inference (1.6 ms latency on T4), Ultralytics-compatible install path. The contribution is data/pipeline-side and doesn't depend on this choice; the pipeline transfers to any modern detector.

2. **"Why not just train at higher `imgsz`?"** — Tried this; GPU memory scales quadratically. Even at `imgsz=1280`, the 4000×3000 image still has to fit, and persons still come out at ~10 pixels — better than 5, but still too small. SAHI at native resolution is strictly better, but only when training matches.

3. **"How is Plan A different from just applying SAHI? Isn't this engineering, not research?"** — The novelty is the **coupling**, not either half alone. Naive SAHI on a baseline model produces 0.255 mAP — *worse* than the baseline 0.759. The two halves are non-trivially interdependent; that's the finding.

4. **"Three negative results — won't reviewers reject this?"** — Plan A is the contribution. The negative results are reported as a methodological lesson, not as the main result. Workshop venues value honest scientific reporting; this framing typically helps rather than hurts.

5. **"What about real-time deployment? You said real time."** — Plan A inference is currently around 3 seconds per 4000×3000 image with SAHI on a T4 — not real-time yet. Real-time engineering is future work; the current contribution is the accuracy ceiling.

6. **"What's the path to fixing the negative results?"** — Each `phase4_*_result.json` file in the repo lists specific knobs: SAM2 needs scale-conditional padding, cascade needs threshold sweeps + 128×128 input, synthetic augmentation needs crop-then-synthesize. None of these are conceptual blockers — they're tuning work.

7. **"Why HERIDAL? Why not SARD or another SAR dataset?"** — HERIDAL is the de-facto wilderness SAR benchmark in the literature. It's the dataset most prior work in this area reports on, so my numbers are comparable. SARD is an alternative I'd use for cross-domain evaluation in a future phase.
