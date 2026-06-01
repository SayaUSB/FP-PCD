import os
import numpy as np
import cv2


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    return (mask > 127).astype(np.uint8) * 255


def list_files(folder: str, ext: str) -> list[str]:
    ext = ext.lower()
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(ext)
    ]
    return sorted(files)
