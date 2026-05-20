"""
Phase 4 Module 1 - SAM2 Bbox Refinement on top of Plan A
========================================================

KAGGLE: paste ENTIRE file into ONE cell. Edit API_KEY line. Run.
Required: GPU T4, Internet ON, attach any Kaggle dataset that contains
the Plan A .pt file. Script auto-discovers it under /kaggle/input/.
"""

# ============================================================
# STEP 0 - INSTALLS  (subprocess so .py and Jupyter both work)
# ============================================================
import subprocess, sys, os, urllib.request

def pip_install(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

print("Installing packages...")
pip_install("ultralytics==8.4.*", "sahi==0.11.*", "pycocotools", "roboflow")
pip_install("git+https://github.com/facebookresearch/sam2.git")

SAM2_CKPT_PATH = "/kaggle/working/sam2_hiera_small.pt"
if not os.path.exists(SAM2_CKPT_PATH):
    print("Downloading SAM2 checkpoint...")
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        SAM2_CKPT_PATH,
    )
print("Install done.\n")

# ============================================================
# STEP 1 - CONFIG
# ============================================================
API_KEY = "PASTE_YOUR_ROBOFLOW_API_KEY_HERE"   # <- edit before running

ROBOFLOW_WORKSPACE = "hung1244s-workspace"
ROBOFLOW_PROJECT   = "heridal-lrbkc-8vnfq"
ROBOFLOW_VERSION   = 1

SAHI_SLICE   = 640
SAHI_OVERLAP = 0.2
SAHI_CONF    = 0.25

MIN_MASK_AREA_PX = 50
AR_MIN, AR_MAX   = 0.2, 5.0
PAD_REFIT_PX     = 1

# ============================================================
# STEP 2 - DEBUG: LIST /kaggle/input/ AND FIND PLAN A WEIGHTS
# ============================================================
import glob, json, time
from pathlib import Path
import numpy as np
import cv2
import torch

OUT = Path("/kaggle/working")
print("=" * 60)
print("Listing /kaggle/input/ contents")
print("=" * 60)
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        full = os.path.join(root, f)
        size_mb = os.path.getsize(full) / 1e6
        print(f"  {full}  ({size_mb:.1f} MB)")
    if not files and not dirs:
        print(f"  (empty: {root})")
print("=" * 60)

_pts = sorted(glob.glob("/kaggle/input/**/*.pt", recursive=True))
if not _pts:
    raise FileNotFoundError(
        "No .pt file under /kaggle/input/. Go to right sidebar -> "
        "'+ Add Input' -> search your Plan A weights dataset -> Add."
    )
_priority = [p for p in _pts if any(k in p.lower() for k in ["heridal", "crops", "best", "plan_a", "yolov12"])]
PLAN_A_WEIGHTS = _priority[0] if _priority else _pts[0]
print(f"\nPlan A weights selected: {PLAN_A_WEIGHTS}\n")

SAM2_CKPT = SAM2_CKPT_PATH
SAM2_CFG  = "sam2_hiera_s.yaml"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}, torch={torch.__version__}\n")

# ============================================================
# STEP 3 - DOWNLOAD HERIDAL VAL VIA ROBOFLOW
# ============================================================
from roboflow import Roboflow
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
dataset = project.version(ROBOFLOW_VERSION).download("coco", location=str(OUT / "heridal"))
print(f"Downloaded HERIDAL to {dataset.location}")

VAL_IMG_DIR = Path(dataset.location) / "valid"
VAL_ANN     = VAL_IMG_DIR / "_annotations.coco.json"
assert VAL_IMG_DIR.exists() and VAL_ANN.exists(), f"Missing {VAL_ANN}"

with open(VAL_ANN) as f:
    coco_val = json.load(f)
print(f"Val: {len(coco_val['images'])} images, {len(coco_val['annotations'])} GT boxes")

# Pick the person category_id from GT (Roboflow exports vary: sometimes 0, sometimes 1)
PERSON_CAT_ID = coco_val["categories"][-1]["id"]  # last category is usually the real class
print(f"GT categories: {[(c['id'], c['name']) for c in coco_val['categories']]}")
print(f"Using PERSON_CAT_ID = {PERSON_CAT_ID}\n")

# ============================================================
# STEP 4 - PLAN A SAHI INFERENCE ON VAL
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
print("Plan A loaded.\n")

baseline_preds = []
all_image_bboxes = {}

t0 = time.time()
for i, img_info in enumerate(coco_val["images"]):
    img_path = VAL_IMG_DIR / img_info["file_name"]
    img_id = img_info["id"]
    result = get_sliced_prediction(
        image=str(img_path),
        detection_model=det_model,
        slice_height=SAHI_SLICE, slice_width=SAHI_SLICE,
        overlap_height_ratio=SAHI_OVERLAP, overlap_width_ratio=SAHI_OVERLAP,
        perform_standard_pred=False,
        postprocess_type="NMS", postprocess_match_threshold=0.5,
        verbose=0,
    )
    bbs = []
    for p in result.object_prediction_list:
        b = p.bbox
        bb = [b.minx, b.miny, b.maxx, b.maxy]
        sc = p.score.value
        bbs.append((bb, sc))
        baseline_preds.append({
            "image_id": img_id, "category_id": PERSON_CAT_ID,
            "bbox": [bb[0], bb[1], bb[2]-bb[0], bb[3]-bb[1]],
            "score": sc,
        })
    all_image_bboxes[img_id] = (img_path, bbs)
    if (i + 1) % 20 == 0:
        print(f"  SAHI {i+1}/{len(coco_val['images'])}  elapsed={time.time()-t0:.0f}s")

with open(OUT / "phase4_planA_predictions_coco.json", "w") as f:
    json.dump(baseline_preds, f)
print(f"Plan A baseline: {len(baseline_preds)} preds across {len(all_image_bboxes)} images\n")

# ============================================================
# STEP 5 - SAM2 REFINEMENT
# ============================================================
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

print("Loading SAM2...")
sam2_model = build_sam2(SAM2_CFG, SAM2_CKPT, device=device)
predictor = SAM2ImagePredictor(sam2_model)
print("SAM2 loaded.\n")

def refine_bbox(image_rgb, bbox_xyxy):
    predictor.set_image(image_rgb)
    box = np.array(bbox_xyxy, dtype=np.float32)
    masks, scores, _ = predictor.predict(box=box, multimask_output=False)
    mask = masks[0].astype(bool)
    if mask.sum() < MIN_MASK_AREA_PX:
        return None
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
    return [max(0, x1 - PAD_REFIT_PX), max(0, y1 - PAD_REFIT_PX),
            x2 + PAD_REFIT_PX, y2 + PAD_REFIT_PX]

refined_preds = []
n_kept, n_dropped = 0, 0
t0 = time.time()
for i, (img_id, (img_path, bbs)) in enumerate(all_image_bboxes.items()):
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    for bbox, score in bbs:
        refit = refine_bbox(img_rgb, bbox)
        if refit is None:
            n_dropped += 1; continue
        x1, y1, x2, y2 = refit
        refined_preds.append({
            "image_id": img_id, "category_id": PERSON_CAT_ID,
            "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
            "score": float(score),
        })
        n_kept += 1
    if (i + 1) % 20 == 0:
        print(f"  SAM2 {i+1}/{len(all_image_bboxes)}  kept={n_kept} dropped={n_dropped}  elapsed={time.time()-t0:.0f}s")

with open(OUT / "phase4_sam2_predictions_coco.json", "w") as f:
    json.dump(refined_preds, f)
total = n_kept + n_dropped
print(f"SAM2: kept={n_kept}, dropped={n_dropped} ({n_dropped/max(total,1)*100:.1f}% filtered)\n")

# ============================================================
# STEP 6 - PYCOCOTOOLS EVAL
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
        "mAP50_95": float(e.stats[0]), "mAP50": float(e.stats[1]),
        "mAP75": float(e.stats[2]), "AP_small": float(e.stats[3]),
        "AP_medium": float(e.stats[4]), "AP_large": float(e.stats[5]),
        "AR_1": float(e.stats[6]), "AR_10": float(e.stats[7]),
        "AR_100": float(e.stats[8]),
    }

print("\n=== Plan A baseline (re-eval) ===")
baseline_metrics = eval_coco(OUT / "phase4_planA_predictions_coco.json", "Plan A (baseline)")
print("\n=== Plan A + SAM2 refinement ===")
sam2_metrics = eval_coco(OUT / "phase4_sam2_predictions_coco.json", "Plan A + SAM2")

with open(OUT / "phase4_sam2_metrics_baseline.json", "w") as f:
    json.dump(baseline_metrics, f, indent=2)
with open(OUT / "phase4_sam2_metrics.json", "w") as f:
    json.dump(sam2_metrics, f, indent=2)

print("\n=== DELTA (SAM2 vs Plan A) ===")
for k in ["mAP50", "mAP50_95", "mAP75", "AP_small", "AR_100"]:
    d = sam2_metrics[k] - baseline_metrics[k]
    print(f"  {k:10s}  {baseline_metrics[k]:.4f}  ->  {sam2_metrics[k]:.4f}   delta {d:+.4f}")

# ============================================================
# STEP 7 - VIZ (10 samples, before/after)
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

VIZ_DIR = OUT / "phase4_sam2_viz"
VIZ_DIR.mkdir(exist_ok=True)

for img_id in list(all_image_bboxes.keys())[:10]:
    img_path, bbs = all_image_bboxes[img_id]
    img_rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax in axes:
        ax.imshow(img_rgb); ax.axis("off")
    axes[0].set_title("Plan A baseline")
    axes[1].set_title("Plan A + SAM2 refinement")
    for bbox, _ in bbs:
        x1, y1, x2, y2 = bbox
        axes[0].add_patch(mpatches.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="lime", linewidth=2))
        refit = refine_bbox(img_rgb, bbox)
        if refit is not None:
            rx1, ry1, rx2, ry2 = refit
            axes[1].add_patch(mpatches.Rectangle((rx1, ry1), rx2-rx1, ry2-ry1, fill=False, edgecolor="lime", linewidth=2))
        else:
            axes[1].add_patch(mpatches.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="red", linewidth=2, linestyle=":"))
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"viz_{img_id}.png", dpi=80, bbox_inches="tight")
    plt.close()

print(f"\nSaved viz to {VIZ_DIR}")
print("\nDONE. Outputs in /kaggle/working/: phase4_sam2_*.json, phase4_sam2_viz/")
