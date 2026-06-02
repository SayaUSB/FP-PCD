import os
import glob
import cv2
import numpy as np
from .utils import read_image, read_mask


def load_drive(root: str, split: str = "training") -> list[dict]:
    """
    DRIVE structure:
      <root>/<split>/images/*.tif (or .png)
      <root>/<split>/mask/*_mask.*
      <root>/<split>/1st_manual/*_manual1.*
      <root>/<split>/2nd_manual/*_manual2.*  (test split only)
    """
    split_dir = os.path.join(root, split)
    img_files = sorted(glob.glob(os.path.join(split_dir, "images", "*")))
    mask_files = sorted(glob.glob(os.path.join(split_dir, "mask", "*")))
    gt1_files = sorted(glob.glob(os.path.join(split_dir, "1st_manual", "*")))
    gt2_dir = os.path.join(split_dir, "2nd_manual")
    gt2_files = sorted(glob.glob(os.path.join(gt2_dir, "*"))) if os.path.isdir(gt2_dir) else []

    samples = []
    for i, img_path in enumerate(img_files):
        name = os.path.splitext(os.path.basename(img_path))[0]
        sample = {
            "name": name,
            "image": read_image(img_path),
            "mask": read_mask(mask_files[i]) if i < len(mask_files) else None,
            "gt1": read_mask(gt1_files[i]) if i < len(gt1_files) else None,
            "gt2": read_mask(gt2_files[i]) if i < len(gt2_files) else None,
        }
        samples.append(sample)
    return samples


def load_stare(root: str) -> list[dict]:
    """
    STARE structure:
      <root>/images/*.ppm (or other format)
      <root>/labels-ah/*.ah.ppm (first annotator)
      <root>/labels-vk/*.vk.ppm (second annotator, optional)
    """
    img_files = sorted(glob.glob(os.path.join(root, "images", "*")))
    ah_files = sorted(glob.glob(os.path.join(root, "labels-ah", "*")))
    vk_dir = os.path.join(root, "labels-vk")
    vk_files = sorted(glob.glob(os.path.join(vk_dir, "*"))) if os.path.isdir(vk_dir) else []

    samples = []
    for i, img_path in enumerate(img_files):
        name = os.path.splitext(os.path.basename(img_path))[0]
        img = read_image(img_path)
        h, w = img.shape[:2]
        sample = {
            "name": name,
            "image": img,
            "mask": _generate_circular_mask(h, w),
            "gt1": read_mask(ah_files[i]) if i < len(ah_files) else None,
            "gt2": read_mask(vk_files[i]) if i < len(vk_files) else None,
        }
        samples.append(sample)
    return samples


def load_chase(root: str) -> list[dict]:
    """
    CHASE_DB1 structure (all files in one flat directory):
      Image_01L.jpg, Image_01L_1stHO.png, Image_01L_2ndHO.png
      Image_01R.jpg, Image_01R_1stHO.png, Image_01R_2ndHO.png ...
    """
    img_files = sorted([
        f for f in glob.glob(os.path.join(root, "*.jpg"))
        if "_HO" not in f
    ])

    samples = []
    for img_path in img_files:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        gt1_path = os.path.join(root, f"{stem}_1stHO.png")
        gt2_path = os.path.join(root, f"{stem}_2ndHO.png")
        img = read_image(img_path)
        h, w = img.shape[:2]
        sample = {
            "name": stem,
            "image": img,
            "mask": _generate_circular_mask(h, w),
            "gt1": read_mask(gt1_path) if os.path.isfile(gt1_path) else None,
            "gt2": read_mask(gt2_path) if os.path.isfile(gt2_path) else None,
        }
        samples.append(sample)
    return samples


def load_aptos(root: str) -> list[dict]:
    """
    APTOS 2019 structure:
      <root>/train_images/*.png
      <root>/train.csv  (columns: id_code, diagnosis 0-4)
    Returns binary label: 0 = normal (grade 0), 1 = abnormal (grade 1-4)
    """
    import pandas as pd
    csv_path = os.path.join(root, "train.csv")
    img_dir = os.path.join(root, "train_images")
    df = pd.read_csv(csv_path)

    samples = []
    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, f"{row['id_code']}.png")
        if not os.path.isfile(img_path):
            continue
        img = read_image(img_path)
        h, w = img.shape[:2]
        samples.append({
            "name": row["id_code"],
            "image": img,
            "mask": _generate_circular_mask(h, w),
            "gt1": None,
            "gt2": None,
            "grade": int(row["diagnosis"]),
            "label": 0 if int(row["diagnosis"]) == 0 else 1,
        })
    return samples


def _generate_circular_mask(h: int, w: int, margin_ratio: float = 0.03) -> np.ndarray:
    """Generate a circular FOV mask for datasets without pre-supplied masks."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) / 2 * (1 - margin_ratio))
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask
