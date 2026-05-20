"""
Phase 4 Module 2 - Hard-Negative Cascade Classifier on top of Plan A
====================================================================

KAGGLE: paste ENTIRE file into ONE cell. Edit API_KEY line. Run.
Required: GPU T4, Internet ON, attach any Kaggle dataset that contains
the Plan A .pt file. Script auto-discovers it under /kaggle/input/.
Expected runtime: 2-3 hours on T4.
"""

# ============================================================
# STEP 0 - INSTALLS
# ============================================================
import subprocess, sys, os

def pip_install(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

print("Installing packages...")
pip_install("ultralytics==8.4.*", "sahi==0.11.*", "pycocotools", "roboflow", "timm")
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

CROP_SCALE = 1.5
CROP_SIZE  = 64
IOU_TP     = 0.5
IOU_FP_MAX = 0.1

CASCADE_THRESHOLD = 0.5
SCORE_COMBINE     = "multiply"

EPOCHS = 30
BATCH  = 64
LR     = 1e-3

# ============================================================
# STEP 2 - DEBUG: LIST /kaggle/input/ AND FIND PLAN A WEIGHTS
# ============================================================
import glob, json, time, random
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

OUT = Path("/kaggle/working")
random.seed(42); np.random.seed(42); torch.manual_seed(42)

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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}, torch={torch.__version__}\n")

# ============================================================
# STEP 3 - DOWNLOAD HERIDAL (train + val)
# ============================================================
assert API_KEY != "PASTE_YOUR_ROBOFLOW_API_KEY_HERE", \
    "Edit API_KEY in STEP 1 with your Roboflow private key first."

from roboflow import Roboflow
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
dataset = project.version(ROBOFLOW_VERSION).download("coco", location=str(OUT / "heridal"))
print(f"Downloaded HERIDAL to {dataset.location}")

TRAIN_IMG_DIR = Path(dataset.location) / "train"
TRAIN_ANN     = TRAIN_IMG_DIR / "_annotations.coco.json"
VAL_IMG_DIR   = Path(dataset.location) / "valid"
VAL_ANN       = VAL_IMG_DIR / "_annotations.coco.json"

with open(TRAIN_ANN) as f: coco_train = json.load(f)
with open(VAL_ANN)   as f: coco_val   = json.load(f)
print(f"Train: {len(coco_train['images'])} images, Val: {len(coco_val['images'])} images\n")

# ============================================================
# STEP 4 - PLAN A SAHI INFERENCE (train + val)
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

def run_sahi(img_dir, coco_dict, tag):
    preds = {}; coco_preds = []
    t0 = time.time()
    for i, img_info in enumerate(coco_dict["images"]):
        img_path = img_dir / img_info["file_name"]
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
            coco_preds.append({
                "image_id": img_info["id"], "category_id": 0,
                "bbox": [bb[0], bb[1], bb[2]-bb[0], bb[3]-bb[1]],
                "score": sc,
            })
        preds[img_info["id"]] = (img_path, bbs)
        if (i + 1) % 50 == 0:
            print(f"  [{tag}] SAHI {i+1}/{len(coco_dict['images'])}  elapsed={time.time()-t0:.0f}s")
    return preds, coco_preds

print("SAHI on TRAIN (~40-50 min)...")
train_preds, _ = run_sahi(TRAIN_IMG_DIR, coco_train, "train")
print("SAHI on VAL (~10-15 min)...")
val_preds, baseline_coco_preds = run_sahi(VAL_IMG_DIR, coco_val, "val")
with open(OUT / "phase4_planA_predictions_coco.json", "w") as f:
    json.dump(baseline_coco_preds, f)

# ============================================================
# STEP 5 - BUILD CASCADE CROPS DATASET FROM TRAIN PREDS vs GT
# ============================================================
def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    return inter / max((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter, 1e-6)

gt_by_img = {}
for ann in coco_train["annotations"]:
    img_id = ann["image_id"]
    x, y, w, h = ann["bbox"]
    gt_by_img.setdefault(img_id, []).append([x, y, x+w, y+h])

DATA_DIR = OUT / "phase4_cascade_dataset"
for split in ["train", "val"]:
    for cls in ["person", "not_person"]:
        (DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)

def save_crop(img, bbox, out_path):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1+x2)/2, (y1+y2)/2
    w, h = (x2-x1) * CROP_SCALE, (y2-y1) * CROP_SCALE
    nx1, ny1 = max(0, int(cx - w/2)), max(0, int(cy - h/2))
    nx2 = min(img.shape[1], int(cx + w/2))
    ny2 = min(img.shape[0], int(cy + h/2))
    if nx2 <= nx1 or ny2 <= ny1: return False
    crop = cv2.resize(img[ny1:ny2, nx1:nx2], (CROP_SIZE, CROP_SIZE))
    cv2.imwrite(str(out_path), crop); return True

n_pos, n_neg, n_ambig = 0, 0, 0
for img_id, (img_path, bbs) in train_preds.items():
    img = cv2.imread(str(img_path))
    gts = gt_by_img.get(img_id, [])
    split = "val" if (img_id % 10 == 0) else "train"
    for j, (bbox, score) in enumerate(bbs):
        iou_max = max((bbox_iou(bbox, g) for g in gts), default=0.0)
        if iou_max >= IOU_TP:
            cls = "person"; n_pos += 1
        elif iou_max < IOU_FP_MAX:
            cls = "not_person"; n_neg += 1
        else:
            n_ambig += 1; continue
        save_crop(img, bbox, DATA_DIR / split / cls / f"{img_id}_{j}.png")

print(f"Cascade dataset: person={n_pos}, not_person={n_neg}, ambig_dropped={n_ambig}\n")

# ============================================================
# STEP 6 - TRAIN MOBILENETV3-SMALL
# ============================================================
import timm

train_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class CascadeDS(Dataset):
    def __init__(self, root, tf):
        self.tf = tf
        self.samples = []
        for cls, label in [("person", 1), ("not_person", 0)]:
            for p in (root / cls).glob("*.png"):
                self.samples.append((p, label))
        random.shuffle(self.samples)
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, y = self.samples[i]
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        return self.tf(img), y

train_ds = CascadeDS(DATA_DIR / "train", train_tf)
val_ds   = CascadeDS(DATA_DIR / "val", val_tf)
print(f"Cascade train={len(train_ds)}, val={len(val_ds)}")

labels = np.array([y for _, y in train_ds.samples])
class_counts = np.bincount(labels)
weights = 1.0 / class_counts[labels]
sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2)

model = timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=2).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
ce = nn.CrossEntropyLoss()

best_val_acc = 0.0
for ep in range(EPOCHS):
    model.train(); tr_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = ce(model(x), y); loss.backward(); opt.step()
        tr_loss += loss.item()
    sched.step()
    model.eval()
    correct = total = tp = fp = fn = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item(); total += y.size(0)
            tp += ((pred == 1) & (y == 1)).sum().item()
            fp += ((pred == 1) & (y == 0)).sum().item()
            fn += ((pred == 0) & (y == 1)).sum().item()
    val_acc = correct / total
    prec = tp / max(tp+fp, 1); rec = tp / max(tp+fn, 1)
    print(f"  ep {ep+1:02d}  loss={tr_loss/len(train_loader):.4f}  val_acc={val_acc:.4f}  P={prec:.3f}  R={rec:.3f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), OUT / "phase4_cascade_classifier.pt")

print(f"Best val acc: {best_val_acc:.4f}\n")
model.load_state_dict(torch.load(OUT / "phase4_cascade_classifier.pt"))

# ============================================================
# STEP 7 - APPLY CASCADE ON VAL PLAN A PREDS
# ============================================================
model.eval()

def cascade_filter(img, bbox, score):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1+x2)/2, (y1+y2)/2
    w, h = (x2-x1) * CROP_SCALE, (y2-y1) * CROP_SCALE
    nx1, ny1 = max(0, int(cx - w/2)), max(0, int(cy - h/2))
    nx2 = min(img.shape[1], int(cx + w/2))
    ny2 = min(img.shape[0], int(cy + h/2))
    if nx2 <= nx1 or ny2 <= ny1: return None, 0.0
    crop = cv2.cvtColor(cv2.resize(img[ny1:ny2, nx1:nx2], (CROP_SIZE, CROP_SIZE)), cv2.COLOR_BGR2RGB)
    x = val_tf(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)[0, 1].item()
    if p < CASCADE_THRESHOLD: return None, p
    new_score = score * p if SCORE_COMBINE == "multiply" else 0.5 * score + 0.5 * p
    return bbox, new_score

cascade_preds = []
n_kept = n_dropped = 0
for img_id, (img_path, bbs) in val_preds.items():
    img = cv2.imread(str(img_path))
    for bbox, score in bbs:
        out_bbox, new_score = cascade_filter(img, bbox, score)
        if out_bbox is None:
            n_dropped += 1; continue
        x1, y1, x2, y2 = out_bbox
        cascade_preds.append({
            "image_id": img_id, "category_id": 0,
            "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
            "score": float(new_score),
        })
        n_kept += 1
with open(OUT / "phase4_cascade_predictions_coco.json", "w") as f:
    json.dump(cascade_preds, f)
print(f"Cascade applied: kept={n_kept}, dropped={n_dropped} ({n_dropped/max(n_kept+n_dropped,1)*100:.1f}% filtered)\n")

# ============================================================
# STEP 8 - PYCOCOTOOLS EVAL
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
        "AR_100": float(e.stats[8]),
    }

print("\n=== Plan A baseline ===")
baseline_metrics = eval_coco(OUT / "phase4_planA_predictions_coco.json", "Plan A")
print("\n=== Plan A + Cascade ===")
cascade_metrics = eval_coco(OUT / "phase4_cascade_predictions_coco.json", "Plan A + Cascade")

with open(OUT / "phase4_cascade_metrics_baseline.json", "w") as f:
    json.dump(baseline_metrics, f, indent=2)
with open(OUT / "phase4_cascade_metrics.json", "w") as f:
    json.dump(cascade_metrics, f, indent=2)

print("\n=== DELTA (Cascade vs Plan A) ===")
for k in ["mAP50", "mAP50_95", "mAP75", "AP_small", "AR_100"]:
    d = cascade_metrics[k] - baseline_metrics[k]
    print(f"  {k:10s}  {baseline_metrics[k]:.4f}  ->  {cascade_metrics[k]:.4f}   delta {d:+.4f}")

print("\nDONE. Outputs in /kaggle/working/: phase4_cascade_*.json, phase4_cascade_classifier.pt")
