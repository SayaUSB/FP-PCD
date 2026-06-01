import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage.morphology import (
    remove_small_objects,
    opening,
    closing,
    disk,
)


def apply_otsu(img: np.ndarray, fov_mask: np.ndarray) -> np.ndarray:
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    roi_pixels = img_norm[fov_mask > 0]
    thresh = threshold_otsu(roi_pixels)
    binary = (img_norm >= thresh).astype(np.uint8) * 255
    binary[fov_mask == 0] = 0
    return binary


def morphological_postprocess(binary: np.ndarray, min_area: int = 50) -> np.ndarray:
    bool_img = binary > 0
    selem = disk(1)
    opened = opening(bool_img, selem)
    closed = closing(opened, selem)
    cleaned = remove_small_objects(closed, max_size=min_area - 1)
    return (cleaned * 255).astype(np.uint8)


def run(
    img: np.ndarray,
    fov_mask: np.ndarray,
    return_intermediates: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    binary = apply_otsu(img, fov_mask)
    result = morphological_postprocess(binary)
    result[fov_mask == 0] = 0
    if return_intermediates:
        return result, {"otsu": binary}
    return result
