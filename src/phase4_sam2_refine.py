"""
Phase 4 Module 1 — SAM2 Bbox Refinement on top of Plan A
=========================================================

KAGGLE SETUP
------------
1. New Notebook, accelerator = GPU T4 x1, Internet ON.
2. Attach Plan A weights dataset: hung1244/yolov12s-heridal-crops-best
3. Edit the API_KEY line below (line ~40) with your Roboflow private key
   (https://app.roboflow.com → Settings → Roboflow API → Private API Key).
4. Paste THIS ENTIRE FILE into one Kaggle cell. Run.
   - Installs happen at the top via subprocess (no separate cell needed).

EXPECTED RUNTIME: ~30-60 minutes on T4 (313 val images, ~3-5 bboxes per image).

OUTPUTS (saved to /kaggle/working/)
-----------------------------------
  - phase4_sam2_predictions_coco.json    (COCO-format predictions, refined)
  - phase4_sam2_metrics.json             (mAP@0.5, mAP@0.5:0.95, etc.)
  - phase4_sam2_metrics_baseline.json    (Plan A baseline, for comparison)
  - phase4_sam2_viz/                     (before/after grid, ~10 images)
"""

# %% ============================================================
# 0. INSTALLS (run inline — works whether pasted into Jupyter cell or run as .py)
# ============================================================
import subprocess, sys, urllib.request, os

def pip_install(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

pip_install("ultralytics==8.4.*", "sahi==0.11.*", "pycocotools", "roboflow")
pip_install("git+https://github.com/facebookresearch/sam2.git")

SAM2_CKPT_PATH = "/kaggle/working/sam2_hiera_small.pt"
if not os.path.exists(SAM2_CKPT_PATH):
    print("Downloading SAM2 checkpoint...")
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        SAM2_CKPT_PATH,
    )

# %% ============================================================
# 1. CONFIG  ← EDIT API_KEY BELOW
# ============================================================
import json, time
from pathlib import Path
import numpy as np
import cv2
import torch

API_KEY = "PASTE_YOUR_ROBOFLOW_API_KEY_HERE"   # ← edit this line in Kaggle before running

OUT = Path("/kaggle/working")
PLAN_A_WEIGHTS = "/kaggle/input/yolov12s-heridal-crops-best/yolov12s_heridal_crops_BEST.pt"
SAM2_CKPT      = SAM2_CKPT_PATH
SAM2_CFG       = "sam2_hiera_s.yaml"

# Roboflow dataset
ROBOFLOW_WORKSPACE = "hung1244s-workspace"
ROBOFLOW_PROJECT   = "heridal-lrbkc-8vnfq"
ROBOFLOW_VERSION   = 1

# SAHI inference config (must match Plan A's evaluation)
SAHI_SLICE = 640
SAHI_OVERLAP = 0.2
SAHI_CONF = 0.25

# SAM2 refinement hyperparams (sweep these manually if first run shows issues)
MIN_MASK_AREA_PX  = 50      # drop if mask < this many pixels (FP filter)
AR_MIN, AR_MAX    = 0.2, 5.0  # aspect ratio bounds for refit bbox
PAD_REFIT_PX      = 1       # 1-pixel outward padding after refit

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}, torch={torch.__version__}")

# %% ============================================================
# 2. DOWNLOAD HERIDAL VAL VIA ROBOFLOW
# ============================================================
assert API_KEY != "PASTE_YOUR_ROBOFLOW_API_KEY_HERE", \
    "Edit API_KEY in the CONFIG cell with your Roboflow private key first."
from roboflow import Roboflow
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
dataset = project.version(ROBOFLOW_VERSION).download("coco", location=str(OUT / "heridal"))
print("Downloaded HERIDAL to", dataset.location)

VAL_IMG_DIR = Path(dataset.location) / "valid"
VAL_ANN     = VAL_IMG_DIR / "_annotations.coco.json"
assert VAL_IMG_DIR.exists() and VAL_ANN.exists()

with open(VAL_ANN) as f:
    coco_val = json.load(f)
print(f"Val: {len(coco_val['images'])} images, {len(coco_val['annotations'])} GT boxes")

# %% ============================================================
# 3. RUN PLAN A SAHI INFERENCE ON VAL  (baseline + bbox prompts for SAM2)
# ============================================================
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

print("Loading Plan A model...")
det_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=PLAN_A_WEIGHTS,
    confidence_threshold=SAHI_CONF,
    device=device,
)

# coco_predictions = list of dicts {image_id, category_id, bbox=[x,y,w,h], score}
baseline_preds = []
all_image_bboxes = {}  # image_id -> list of (bbox_xyxy, score)

t0 = time.time()
for i, img_info in enumerate(coco_val["images"]):
    img_path = VAL_IMG_DIR / img_info["file_name"]
    img = cv2.imread(str(img_path))
    img_id = img_info["id"]

    result = get_sliced_prediction(
        image=str(img_path),
        detection_model=det_model,
        slice_height=SAHI_SLICE,
        slice_width=SAHI_SLICE,
        overlap_height_ratio=SAHI_OVERLAP,
        overlap_width_ratio=SAHI_OVERLAP,
        perform_standard_pred=False,
        postprocess_type="NMS",
        postprocess_match_threshold=0.5,
        verbose=0,
    )
    bboxes_xyxy_score = []
    for pred in result.object_prediction_list:
        b = pred.bbox
        x1, y1, x2, y2 = b.minx, b.miny, b.maxx, b.maxy
        score = pred.score.value
        bboxes_xyxy_score.append(([x1, y1, x2, y2], score))
        baseline_preds.append({
            "image_id": img_id,
            "category_id": 0,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": score,
        })
    all_image_bboxes[img_id] = (img_path, bboxes_xyxy_score)
    if (i + 1) % 20 == 0:
        print(f"  SAHI {i+1}/{len(coco_val['images'])}  elapsed={time.time()-t0:.1f}s")

with open(OUT / "phase4_planA_predictions_coco.json", "w") as f:
    json.dump(baseline_preds, f)
print(f"Plan A baseline: {len(baseline_preds)} preds across {len(all_image_bboxes)} images")

# %% ============================================================
# 4. SAM2 REFINEMENT
# ============================================================
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

print("Loading SAM2...")
sam2_model = build_sam2(SAM2_CFG, SAM2_CKPT, device=device)
predictor = SAM2ImagePredictor(sam2_model)


def refine_bbox(image_rgb, bbox_xyxy):
    """Returns refit bbox [x1,y1,x2,y2] or None if filter drops it."""
    predictor.set_image(image_rgb)
    box = np.array(bbox_xyxy, dtype=np.float32)
    masks, scores, _ = predictor.predict(box=box, multimask_output=False)
    mask = masks[0].astype(bool)
    if mask.sum() < MIN_MASK_AREA_PX:
        return None
    # largest connected component
    n_cc, lab = cv2.connectedComponents(mask.astype(np.uint8))
    if n_cc > 2:
        sizes = [(lab == k).sum() for k in range(1, n_cc)]
        keep = 1 + int(np.argmax(sizes))
        mask = (lab == keep)
        if mask.sum() < MIN_MASK_AREA_PX:
            return None
    ys, xs = np.where(mask)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    w, h = x2 - x1, y2 - y1
    if w == 0 or h == 0:
        return None
    ar = w / h
    if ar < AR_MIN or ar > AR_MAX:
        return None
    return [
        max(0, x1 - PAD_REFIT_PX),
        max(0, y1 - PAD_REFIT_PX),
        x2 + PAD_REFIT_PX,
        y2 + PAD_REFIT_PX,
    ]


refined_preds = []
n_kept, n_dropped = 0, 0
t0 = time.time()
for i, (img_id, (img_path, bboxes_xyxy_score)) in enumerate(all_image_bboxes.items()):
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    for bbox_xyxy, score in bboxes_xyxy_score:
        refit = refine_bbox(img_rgb, bbox_xyxy)
        if refit is None:
            n_dropped += 1
            continue
        x1, y1, x2, y2 = refit
        refined_preds.append({
            "image_id": img_id,
            "category_id": 0,
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(score),
        })
        n_kept += 1
    if (i + 1) % 20 == 0:
        print(f"  SAM2 {i+1}/{len(all_image_bboxes)}  kept={n_kept} dropped={n_dropped}  elapsed={time.time()-t0:.1f}s")

with open(OUT / "phase4_sam2_predictions_coco.json", "w") as f:
    json.dump(refined_preds, f)
print(f"SAM2 refinement: kept={n_kept}, dropped={n_dropped} ({n_dropped/(n_kept+n_dropped)*100:.1f}% filtered)")

# %% ============================================================
# 5. PYCOCOTOOLS EVALUATION
# ============================================================
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def eval_coco(pred_path, name):
    coco_gt = COCO(str(VAL_ANN))
    coco_dt = coco_gt.loadRes(str(pred_path))
    e = COCOeval(coco_gt, coco_dt, iouType="bbox")
    e.evaluate(); e.accumulate(); e.summarize()
    return {
        "name": name,
        "mAP50_95": float(e.stats[0]),
        "mAP50":    float(e.stats[1]),
        "mAP75":    float(e.stats[2]),
        "AP_small": float(e.stats[3]),
        "AP_medium":float(e.stats[4]),
        "AP_large": float(e.stats[5]),
        "AR_1":     float(e.stats[6]),
        "AR_10":    float(e.stats[7]),
        "AR_100":   float(e.stats[8]),
    }

print("\n=== Plan A baseline (re-eval) ===")
baseline_metrics = eval_coco(OUT / "phase4_planA_predictions_coco.json", "Plan A (baseline)")
print("\n=== Plan A + SAM2 refinement ===")
sam2_metrics = eval_coco(OUT / "phase4_sam2_predictions_coco.json", "Plan A + SAM2")

with open(OUT / "phase4_sam2_metrics_baseline.json", "w") as f:
    json.dump(baseline_metrics, f, indent=2)
with open(OUT / "phase4_sam2_metrics.json", "w") as f:
    json.dump(sam2_metrics, f, indent=2)

# Delta summary
print("\n=== DELTA (SAM2 vs Plan A) ===")
for k in ["mAP50", "mAP50_95", "mAP75", "AP_small", "AR_100"]:
    d = sam2_metrics[k] - baseline_metrics[k]
    print(f"  {k:10s}  {baseline_metrics[k]:.4f}  →  {sam2_metrics[k]:.4f}   Δ {d:+.4f}")

# %% ============================================================
# 6. VISUALIZATIONS (10 sample images, before/after)
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

VIZ_DIR = OUT / "phase4_sam2_viz"
VIZ_DIR.mkdir(exist_ok=True)

sample_ids = list(all_image_bboxes.keys())[:10]
for img_id in sample_ids:
    img_path, bboxes_xyxy_score = all_image_bboxes[img_id]
    img_rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax in axes:
        ax.imshow(img_rgb); ax.axis("off")
    axes[0].set_title("Plan A baseline")
    axes[1].set_title("Plan A + SAM2 refinement")

    for bbox_xyxy, _ in bboxes_xyxy_score:
        x1, y1, x2, y2 = bbox_xyxy
        axes[0].add_patch(mpatches.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="lime", linewidth=2))
        refit = refine_bbox(img_rgb, bbox_xyxy)
        if refit is not None:
            rx1, ry1, rx2, ry2 = refit
            axes[1].add_patch(mpatches.Rectangle((rx1, ry1), rx2-rx1, ry2-ry1, fill=False, edgecolor="lime", linewidth=2))
        else:
            x1, y1, x2, y2 = bbox_xyxy
            axes[1].add_patch(mpatches.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="red", linewidth=2, linestyle=":"))

    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"viz_{img_id}.png", dpi=80, bbox_inches="tight")
    plt.close()
print(f"Saved {len(sample_ids)} viz to {VIZ_DIR}")
print("\nDONE. Download phase4_sam2_*.json + phase4_sam2_viz/ from /kaggle/working/")
