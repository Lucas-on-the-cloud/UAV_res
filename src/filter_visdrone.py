"""
Filter VisDrone-DET dataset to person-only YOLO format.

Keeps original VisDrone class 0 (pedestrian) and class 1 (people),
remaps both to a single class `person` (id 0).

Usage on Kaggle:
    SOURCE = "/kaggle/input/datasets/banuprasadb/visdrone-dataset/VisDrone_Dataset"
    TARGET = "/kaggle/working/visdrone_person"
    python filter_visdrone.py
"""
import os
import glob

from tqdm import tqdm


SOURCE = "/kaggle/input/datasets/banuprasadb/visdrone-dataset/VisDrone_Dataset"
TARGET = "/kaggle/working/visdrone_person"

# VisDrone class id → new class id. Both pedestrian (0) and people (1) → person (0).
KEEP_CLASSES = {0: 0, 1: 0}

SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val":   "VisDrone2019-DET-val",
}


def filter_split(split_name: str, src_folder: str) -> tuple[int, int]:
    src_labels = f"{SOURCE}/{src_folder}/labels"
    src_images = f"{SOURCE}/{src_folder}/images"
    dst_labels = f"{TARGET}/{split_name}/labels"
    dst_images = f"{TARGET}/{split_name}/images"
    os.makedirs(dst_labels, exist_ok=True)
    os.makedirs(dst_images, exist_ok=True)

    n_kept_img = 0
    n_kept_box = 0

    for lbl_path in tqdm(sorted(glob.glob(f"{src_labels}/*.txt")), desc=split_name):
        kept = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                cls = int(parts[0])
                if cls in KEEP_CLASSES:
                    kept.append(f"{KEEP_CLASSES[cls]} " + " ".join(parts[1:]))
                    n_kept_box += 1

        if not kept:
            continue

        stem = os.path.splitext(os.path.basename(lbl_path))[0]
        with open(f"{dst_labels}/{stem}.txt", "w") as f:
            f.write("\n".join(kept) + "\n")

        for ext in [".jpg", ".JPG", ".jpeg", ".png"]:
            src_img = f"{src_images}/{stem}{ext}"
            if os.path.exists(src_img):
                dst_img = f"{dst_images}/{stem}{ext}"
                if not os.path.exists(dst_img):
                    os.symlink(src_img, dst_img)
                n_kept_img += 1
                break

    return n_kept_img, n_kept_box


def write_yaml() -> None:
    yaml_content = (
        f"path: {TARGET}\n"
        f"train: train/images\n"
        f"val: val/images\n\n"
        f"nc: 1\n"
        f"names:\n  0: person\n"
    )
    with open(f"{TARGET}/visdrone_person.yaml", "w") as f:
        f.write(yaml_content)


def main() -> None:
    for split, folder in SPLITS.items():
        n_img, n_box = filter_split(split, folder)
        print(f"{split}: {n_img} images, {n_box} person boxes")

    write_yaml()
    print(f"Done. Output: {TARGET}")


if __name__ == "__main__":
    main()
