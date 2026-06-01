import numpy as np
import pytest
import cv2

@pytest.fixture
def synthetic_rgb():
    """256x256 RGB fundus-like image: bright circle on black."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(img, (128, 128), 100, (180, 150, 120), -1)
    # Add a dark horizontal vessel
    img[120:125, 30:220] = (100, 80, 60)
    return img

@pytest.fixture
def synthetic_mask():
    """256x256 binary FOV mask."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(mask, (128, 128), 100, 255, -1)
    return mask

@pytest.fixture
def synthetic_gt():
    """256x256 ground truth: horizontal vessel at rows 120-125."""
    gt = np.zeros((256, 256), dtype=np.uint8)
    gt[120:125, 30:220] = 255
    return gt
