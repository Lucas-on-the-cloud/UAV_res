# Phase 4 — Kaggle Setup Instructions

Two Kaggle notebooks run **in parallel** on independent T4 GPUs. Free tier gives 30 GPU-hours/week — these two together use ~5 hours.

## Common prerequisites

1. Have a Kaggle account with **2 simultaneous notebook sessions** allowed (default for free tier).
2. Add Roboflow API key as a Kaggle Secret:
   - Kaggle → user profile → *Add-ons* → *Secrets* → New secret
   - Label: `ROBOFLOW_API_KEY`
   - Value: from https://app.roboflow.com → Settings → Roboflow API → Private API Key
3. The Plan A weights are on Kaggle dataset `hung1244/yolov12s-heridal-crops-best` — attach it (Add Input → search by username).

---

## Notebook 1 — Module 1 (SAM2 Refinement)

### Setup

1. Kaggle → Create → New Notebook
2. Sidebar:
   - **Accelerator:** GPU T4 x1
   - **Internet:** ON
   - **Add Input:** `hung1244/yolov12s-heridal-crops-best`
3. First cell — installs:
   ```bash
   !pip install -q ultralytics==8.4.* sahi==0.11.* pycocotools roboflow
   !pip install -q git+https://github.com/facebookresearch/sam2.git
   !wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt -O /kaggle/working/sam2_hiera_small.pt
   ```
4. Second cell — copy contents of [`src/phase4_sam2_refine.py`](../src/phase4_sam2_refine.py) (skip the install lines at top, they're already done).
5. Run all. Save Version when finished.

### Expected outputs (in `/kaggle/working/`)
- `phase4_sam2_predictions_coco.json`
- `phase4_sam2_metrics.json`
- `phase4_sam2_metrics_baseline.json`
- `phase4_sam2_viz/viz_*.png` (10 before/after images)

### Expected runtime
- SAHI on 313 val images: ~10-15 min
- SAM2 refinement: ~20-30 min
- Eval + viz: ~5 min
- **Total: 40-60 min**

---

## Notebook 2 — Module 2 (Cascade Classifier)

### Setup

1. **Different browser tab / Kaggle session** from Notebook 1 (so they run in parallel).
2. Create → New Notebook.
3. Sidebar:
   - **Accelerator:** GPU T4 x1
   - **Internet:** ON
   - **Add Input:** `hung1244/yolov12s-heridal-crops-best`
4. First cell — installs:
   ```bash
   !pip install -q ultralytics==8.4.* sahi==0.11.* pycocotools roboflow timm
   ```
5. Second cell — copy contents of [`src/phase4_cascade.py`](../src/phase4_cascade.py) (skip install line at top).
6. Run all. Save Version when finished.

### Expected outputs (in `/kaggle/working/`)
- `phase4_cascade_predictions_coco.json`
- `phase4_cascade_metrics.json`
- `phase4_cascade_metrics_baseline.json`
- `phase4_cascade_classifier.pt` (~5 MB MobileNetV3-Small weights)
- `phase4_cascade_dataset/` (training crops — large, can delete after)

### Expected runtime
- SAHI on train (1124 images): ~40-50 min
- SAHI on val (313 images): ~10-15 min
- Cascade dataset crops: ~5 min
- MobileNetV3 training (30 epochs): ~30-60 min
- Apply on val + eval: ~10 min
- **Total: ~2-3 hours**

---

## After both finish

Download the following from each notebook's `/kaggle/working/` (Output panel → right-click each file → Download):

**From Notebook 1:**
- `phase4_sam2_predictions_coco.json`
- `phase4_sam2_metrics.json`
- `phase4_sam2_metrics_baseline.json`

**From Notebook 2:**
- `phase4_cascade_predictions_coco.json`
- `phase4_cascade_metrics.json`
- `phase4_cascade_classifier.pt`

Save them locally to `c:\022026materials\AI\repo_to_push\results\` (new files: `phase4_sam2_*.json`, `phase4_cascade_*.json`). Then ping me with the metrics — I'll write the analysis doc, update the README headline table, and prep the combined-pipeline ablation (Module 3 from the plan).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `SAM2 ModuleNotFoundError` after pip install | sam2 needs torch ≥ 2.3 | Kaggle T4 has torch 2.x; usually fine. If error, try `!pip install torch==2.4.0 --upgrade` then re-install sam2 |
| `Roboflow 401 unauthorized` | API key secret typo | Re-check secret label exactly `ROBOFLOW_API_KEY` (case-sensitive) |
| OOM during SAM2 | image too large, SAM2 holds full embedding | Lower batch is N/A here (one image at a time); fall back to `sam2_hiera_tiny.pt` |
| Cascade `WeightedRandomSampler` very slow | label list huge | Reduce `CROP_SCALE` or set `replacement=False` |
| pycocotools eval gives 0 | image_id / category_id mismatch | Print first few preds + first few GT to verify they use the same id space |
| SAHI very slow (> 1 min/image) | tile size wrong | Confirm `slice_height=slice_width=640` |

Tag me with the error message if anything breaks — I'll patch the script and you re-run from the failing cell.
