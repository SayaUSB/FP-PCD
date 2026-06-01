import numpy as np
import pytest


@pytest.fixture
def small_sample():
    rng = np.random.default_rng(0)
    img_rgb = rng.integers(50, 200, (64, 64, 3), dtype=np.uint8)
    vessel_enhanced = rng.random((64, 64)).astype(np.float32)
    fov_mask = np.full((64, 64), 255, dtype=np.uint8)
    gt = np.zeros((64, 64), dtype=np.uint8)
    gt[20:30, 20:40] = 255
    return img_rgb, vessel_enhanced, fov_mask, gt


def test_extract_features_shape(small_sample):
    from retina_seg.ml.pipeline import extract_features
    img_rgb, vessel_enhanced, fov_mask, _ = small_sample
    features = extract_features(img_rgb, vessel_enhanced, fov_mask)
    n_pixels = int(np.sum(fov_mask > 0))
    assert features.shape == (n_pixels, 5)
    assert features.dtype == np.float32


def test_train_and_predict_svm(small_sample):
    from retina_seg.ml.pipeline import extract_features, train, predict
    img_rgb, vessel_enhanced, fov_mask, gt = small_sample
    features = extract_features(img_rgb, vessel_enhanced, fov_mask)
    labels = (gt[fov_mask > 0] > 0).astype(int)
    model = train(features, labels, method="rf")  # RF is faster for tests
    pred_mask = predict(model, img_rgb, vessel_enhanced, fov_mask)
    assert pred_mask.shape == (64, 64)
    assert pred_mask.dtype == np.uint8
    assert set(np.unique(pred_mask)).issubset({0, 255})
