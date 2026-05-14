# Phase 3 — Pivot from AirSim to Diffusion-based Synthetic Data

**Date:** 2026-05-15
**Status:** In progress

## Context

The original Phase 3 plan was to use the **AirSim** drone simulator to generate photo-realistic synthetic training data, addressing the false-positive failure modes identified in Plan A (model confusing rocks, buildings, and dense vegetation with persons). After substantial effort attempting to set up AirSim on a rented Windows GPU VM (RTX 5090), I switched to a **diffusion-based synthetic data approach** that is better suited to the available compute and the project's timeline.

## Why the pivot

### AirSim setup attempts

1. **Linux pre-built binary downloaded by mistake.** AirSim v1.8.1 publishes Linux and Windows binaries under separate release tags. The first download was the Linux version, unusable on the Windows VM.

2. **Windows binary downloaded and ran successfully.** AirSim's LandscapeMountains environment loaded and rendered correctly on the RTX 5090.

3. **Python client (`airsim` package) failed.** The `airsim` PyPI package depends on `msgpack-rpc-python`, which is abandoned (last updated 2014) and incompatible with Tornado 5+ and Python 3.11+:
   - First fix attempt: install correct dependency order with manual `--no-deps` flags. Got `airsim` importable.
   - Second issue: `msgpackrpc.loop.PeriodicCallback(callback, callback_time, self._ioloop)` — old positional arg for `io_loop` is interpreted as `jitter` in Tornado 5+. Patched by removing the third arg.
   - Third issue: `IOStream(socket, io_loop=...)` — `io_loop` keyword removed in Tornado 5. Patched globally across `msgpackrpc` package.
   - Subsequent issues continued cascading from Tornado API changes.

### Decision: pivot to Stable Diffusion XL

Rather than continue patching a 12-year-old dependency, the better use of the RTX 5090 32 GB compute is **diffusion-based synthetic data generation**, which is:

- A current research direction (2024–2026) with strong recent literature.
- A more efficient match for the RTX 5090 hardware (originally designed for diffusion-style workloads).
- Significantly faster to set up (~1 hour vs days for full AirSim source build).
- Just as photorealistic (SDXL produces state-of-the-art image quality).

## New Phase 3 pipeline

1. **Generate aerial backgrounds** with Stable Diffusion XL (SDXL):
   - Prompts: "aerial drone view of rocky mountain forest, top-down photography, detailed terrain"
   - ControlNet depth-conditioning (optional) for consistent aerial perspective
   - Target: ~2,000 photo-realistic backgrounds
2. **Composite HERIDAL person crops** onto the generated backgrounds:
   - Reuse the cropping pipeline from Plan A
   - Compositing yields exact bounding-box annotations
3. **Refine annotations with SAM2** (Segment Anything Model 2) if needed.
4. **Train YOLOv12-s** on HERIDAL + 2,000 diffusion-synthetic images.
5. **Evaluate with SAHI** on the original 4000×3000 HERIDAL val set, comparing against Plan A.

## Key references

- Azizi et al. (2023). *Synthetic Data from Diffusion Models Improves ImageNet Classification.* arXiv:2304.08466.
- Zhang et al. (2023). *Adding Conditional Control to Text-to-Image Diffusion Models* (ControlNet). ICCV. arXiv:2302.05543.
- Ravi et al. (2024). *SAM 2: Segment Anything in Images and Videos.* arXiv:2408.00714.
- Wu et al. (2023). *DiffuMask: Synthesizing Images with Pixel-level Annotations for Semantic Segmentation Using Diffusion Models.* ICCV. arXiv:2303.11681.

## Implications

- The original "ML + Simulator" framing of the project is reframed as "ML + Generative AI for Synthetic Data." This is consistent with the advisor's intent (synthetic data augmentation) while taking advantage of more modern tooling.
- The AirSim attempt and its failure modes are documented here as a negative result that motivates the diffusion approach — useful as a contrast in the eventual report.

## What stays the same

- Plan A (density-aware crops + SAHI) remains the working baseline (mAP@0.5 = 0.872, mAP@0.5:0.95 = 0.574).
- Phase 3 still targets the same failure modes observed in Plan A (false positives on rocks, buildings, vegetation).
- The evaluation protocol is unchanged: COCO-style mAP on the original 4000×3000 HERIDAL val set, using pycocotools.
