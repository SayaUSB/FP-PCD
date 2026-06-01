import numpy as np
import pytest
import cv2


@pytest.fixture
def vessel_enhanced():
    """128x128 float32 where vessel region has high values."""
    img = np.zeros((128, 128), dtype=np.float32)
    img[:] = 5.0                  # background
    img[60:66, 20:108] = 50.0     # vessel (high response)
    return img


@pytest.fixture
def fov_mask():
    mask = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(mask, (64, 64), 60, 255, -1)
    return mask


def test_apply_otsu_binary_values(vessel_enhanced, fov_mask):
    from retina_seg.segmentation.pipeline import apply_otsu
    result = apply_otsu(vessel_enhanced, fov_mask)
    unique = set(np.unique(result))
    assert unique.issubset({0, 255})
    assert result.shape == vessel_enhanced.shape


def test_apply_otsu_detects_vessel(vessel_enhanced, fov_mask):
    from retina_seg.segmentation.pipeline import apply_otsu
    result = apply_otsu(vessel_enhanced, fov_mask)
    vessel_pixels = result[62, 20:108]
    assert np.all(vessel_pixels == 255), "vessel region should be segmented"


def test_morphological_postprocess_removes_noise(fov_mask):
    from retina_seg.segmentation.pipeline import morphological_postprocess
    binary = np.zeros((128, 128), dtype=np.uint8)
    # single small pixel (noise)
    binary[30, 30] = 255
    # vessel-like line
    binary[60:66, 20:108] = 255
    result = morphological_postprocess(binary, min_area=50)
    assert result[30, 30] == 0, "single pixel noise should be removed"
    assert np.any(result[60:66, 20:108] > 0), "vessel should survive"


def test_run_output_shape_and_dtype(vessel_enhanced, fov_mask):
    from retina_seg.segmentation.pipeline import run
    result = run(vessel_enhanced, fov_mask)
    assert result.shape == vessel_enhanced.shape
    assert result.dtype == np.uint8
    assert set(np.unique(result)).issubset({0, 255})


def test_run_zeros_outside_fov(vessel_enhanced, fov_mask):
    from retina_seg.segmentation.pipeline import run
    result = run(vessel_enhanced, fov_mask)
    outside = result[fov_mask == 0]
    assert np.all(outside == 0), "pixels outside FOV must be 0"
