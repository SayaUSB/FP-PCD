import numpy as np
import cv2
from scipy.ndimage import rotate as ndimage_rotate


def _build_mf_kernel(sigma: float = 2.0, L: int = 9) -> np.ndarray:
    """
    Build a square matched filter kernel for detecting a horizontal dark vessel.
    Cross-section profile: -exp(-row_offset²/2σ²), active for |col_offset| ≤ L//2.
    Mean-subtracted so uniform backgrounds give zero response.
    """
    half = max(int(np.ceil(3 * sigma)), L // 2)
    size = 2 * half + 1
    center = half
    rows, cols = np.mgrid[0:size, 0:size]
    row_off = (rows - center).astype(np.float64)
    col_off = (cols - center).astype(np.float64)
    kernel = np.where(
        np.abs(col_off) <= L // 2,
        -np.exp(-row_off ** 2 / (2.0 * sigma ** 2)),
        0.0,
    )
    kernel -= kernel.mean()
    return kernel.astype(np.float32)


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(img)


def apply_gaussian(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    ksize = int(6 * sigma + 1) | 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def apply_matched_filter(
    img: np.ndarray,
    sigma: float = 1.5,
    L: int = 15,
    n_angles: int = 12,
) -> np.ndarray:
    base = _build_mf_kernel(sigma, L)
    img_f = img.astype(np.float32)
    response = np.full_like(img_f, -np.inf)
    for k in range(n_angles):
        angle = k * 180.0 / n_angles
        rotated = ndimage_rotate(base, angle, reshape=False, order=3).astype(np.float32)
        filtered = cv2.filter2D(img_f, -1, rotated)
        response = np.maximum(response, filtered)
    return response


def run(
    img: np.ndarray,
    return_intermediates: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    clahe_out = apply_clahe(img)
    gauss_out = apply_gaussian(clahe_out)
    enhanced = apply_matched_filter(gauss_out)
    if return_intermediates:
        return enhanced, {"clahe": clahe_out, "gaussian": gauss_out}
    return enhanced
