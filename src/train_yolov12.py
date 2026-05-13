"""
Train YOLOv12 on the filtered VisDrone person dataset.

Designed to be runnable as a single Kaggle cell (commit-and-run mode)
or as a local script.

Usage:
    python train_yolov12.py
"""
import os
import json
import shutil
import sys
import subprocess
import urllib.request


# ==================== CONFIG ====================
MODEL_VARIANT = "yolov12n"   # one of: yolov12n / yolov12s / yolov12m / yolov12l / yolov12x
DATA_YAML = "/kaggle/working/visdrone_person/visdrone_person.yaml"
PROJECT_DIR = "/kaggle/working/runs"
RUN_NAME = f"{MODEL_VARIANT}_visdrone_person"

EPOCHS = 80
IMGSZ = 640
BATCH = 16 if MODEL_VARIANT == "yolov12n" else 12  # smaller batch for larger models
LR0 = 0.001
PATIENCE = 15
# ================================================


def install_ultralytics() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "ultralytics"],
        check=False,
    )


def download_pretrained() -> str:
    url = (
        "https://github.com/sunsmarterjie/yolov12/releases/download/turbo/"
        f"{MODEL_VARIANT}.pt"
    )
    weight = f"/kaggle/working/{MODEL_VARIANT}.pt"
    if not os.path.exists(weight) or os.path.getsize(weight) < 5e6:
        urllib.request.urlretrieve(url, weight)
    return weight


def main() -> None:
    install_ultralytics()
    weight = download_pretrained()
    print(f"Pretrained weight: {weight} ({os.path.getsize(weight)/1e6:.1f} MB)")

    from ultralytics import YOLO

    model = YOLO(weight)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=0,
        patience=PATIENCE,
        save_period=10,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=LR0,
        cos_lr=True,
        amp=True,
        workers=4,
        seed=42,
        plots=True,
    )

    # Backup weights to /kaggle/working/ root for download convenience
    best_src = f"{PROJECT_DIR}/{RUN_NAME}/weights/best.pt"
    last_src = f"{PROJECT_DIR}/{RUN_NAME}/weights/last.pt"
    shutil.copy(best_src, f"/kaggle/working/{MODEL_VARIANT}_visdrone_BEST.pt")
    shutil.copy(last_src, f"/kaggle/working/{MODEL_VARIANT}_visdrone_LAST.pt")

    metrics = model.val()
    summary = {
        "model": MODEL_VARIANT,
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "mAP50":    float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall":    float(metrics.box.mr),
    }
    out_json = f"/kaggle/working/{MODEL_VARIANT}_visdrone_baseline_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("\nIMPORTANT: now click 'Save Version' → 'Quick Save' to persist /kaggle/working/")


if __name__ == "__main__":
    main()
