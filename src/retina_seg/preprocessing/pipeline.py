import numpy as np
import cv2


def extract_green(img_rgb: np.ndarray) -> np.ndarray:
    return img_rgb[:, :, 1]


def generate_fov_mask(img_rgb: np.ndarray, threshold: int = 20) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
    return mask


def correct_illumination(img_green: np.ndarray, radius: int = 70) -> np.ndarray:
    diam = 2 * radius + 1
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diam, diam))
    background = cv2.morphologyEx(img_green, cv2.MORPH_OPEN, disk)
    diff = img_green.astype(np.float32) - background.astype(np.float32)
    corrected = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return corrected.astype(np.uint8)


def run(
    img_rgb: np.ndarray,
    fov_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    green = extract_green(img_rgb)
    if fov_mask is None:
        fov_mask = generate_fov_mask(img_rgb)
    corrected = correct_illumination(green)
    inside = corrected[fov_mask > 0]
    fill_val = int(inside.mean()) if inside.size > 0 else 0
    result = corrected.copy()
    result[fov_mask == 0] = fill_val
    return result, fov_mask
