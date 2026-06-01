import numpy as np
import pytest
import cv2


@pytest.fixture
def dark_vessel_image():
    """128x128 grayscale image with dark horizontal vessel at rows 60-65."""
    img = np.full((128, 128), 180, dtype=np.uint8)
    img[60:66, 20:108] = 80  # dark vessel
    return img


def test_build_mf_kernel_sums_to_zero():
    from retina_seg.enhancement.pipeline import _build_mf_kernel
    kernel = _build_mf_kernel(sigma=2.0, L=9)
    assert abs(kernel.sum()) < 1e-3


def test_build_mf_kernel_shape_is_square():
    from retina_seg.enhancement.pipeline import _build_mf_kernel
    kernel = _build_mf_kernel(sigma=2.0, L=9)
    assert kernel.ndim == 2
    assert kernel.shape[0] == kernel.shape[1]


def test_apply_matched_filter_higher_at_vessel(dark_vessel_image):
    from retina_seg.enhancement.pipeline import apply_matched_filter
    response = apply_matched_filter(dark_vessel_image, sigma=2.0, L=9, n_angles=12)
    vessel_response = response[62, 64]      # center of vessel
    bg_response = response[10, 64]          # background pixel
    assert vessel_response > bg_response


def test_apply_clahe_output_in_range(dark_vessel_image):
    from retina_seg.enhancement.pipeline import apply_clahe
    result = apply_clahe(dark_vessel_image)
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255


def test_run_returns_float32(dark_vessel_image):
    from retina_seg.enhancement.pipeline import run
    result = run(dark_vessel_image)
    assert result.dtype == np.float32
    assert result.shape == dark_vessel_image.shape


def test_run_with_intermediates_returns_dict(dark_vessel_image):
    from retina_seg.enhancement.pipeline import run
    result, intermediates = run(dark_vessel_image, return_intermediates=True)
    assert isinstance(intermediates, dict)
    assert "clahe" in intermediates
    assert "gaussian" in intermediates
