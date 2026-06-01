import numpy as np
import pytest
import cv2


@pytest.fixture
def synthetic_rgb():
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(img, (128, 128), 100, (180, 150, 120), -1)
    img[120:125, 30:220] = (80, 60, 50)  # dark vessel
    return img


@pytest.fixture
def synthetic_mask():
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(mask, (128, 128), 100, 255, -1)
    return mask


def test_extract_green_shape(synthetic_rgb):
    from retina_seg.preprocessing.pipeline import extract_green
    result = extract_green(synthetic_rgb)
    assert result.shape == (256, 256)
    assert result.dtype == np.uint8


def test_generate_fov_mask_nonzero(synthetic_rgb):
    from retina_seg.preprocessing.pipeline import generate_fov_mask
    mask = generate_fov_mask(synthetic_rgb)
    assert mask.shape == (256, 256)
    assert np.any(mask > 0)
    assert set(np.unique(mask)).issubset({0, 255})


def test_correct_illumination_same_shape(synthetic_rgb):
    from retina_seg.preprocessing.pipeline import extract_green, correct_illumination
    green = extract_green(synthetic_rgb)
    corrected = correct_illumination(green)
    assert corrected.shape == green.shape
    assert corrected.dtype == np.uint8


def test_run_returns_tuple_correct_shapes(synthetic_rgb, synthetic_mask):
    from retina_seg.preprocessing.pipeline import run
    corrected, mask = run(synthetic_rgb, synthetic_mask)
    assert corrected.shape == (256, 256)
    assert corrected.dtype == np.uint8
    assert mask.shape == (256, 256)


def test_run_without_mask_generates_mask(synthetic_rgb):
    from retina_seg.preprocessing.pipeline import run
    corrected, mask = run(synthetic_rgb)
    assert np.any(mask > 0)
