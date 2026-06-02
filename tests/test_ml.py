import numpy as np
import pytest


@pytest.fixture
def vessel_sample():
    rng = np.random.default_rng(0)
    vessel_mask = np.zeros((64, 64), dtype=np.uint8)
    vessel_mask[20:30, 10:50] = 255
    vessel_mask[10:50, 30:35] = 255
    fov_mask = np.full((64, 64), 255, dtype=np.uint8)
    return vessel_mask, fov_mask


def test_extract_image_features_shape(vessel_sample):
    from retina_seg.ml.pipeline import extract_image_features, FEATURE_NAMES
    vessel_mask, fov_mask = vessel_sample
    feats = extract_image_features(vessel_mask, fov_mask)
    assert feats.shape == (len(FEATURE_NAMES),)
    assert feats.dtype == np.float32


def test_train_and_predict(vessel_sample):
    from retina_seg.ml.pipeline import extract_image_features, train, predict
    vessel_mask, fov_mask = vessel_sample

    # Build tiny dataset: 10 normal + 10 abnormal
    rng = np.random.default_rng(1)
    features, labels = [], []
    for _ in range(10):
        m = np.zeros((64, 64), dtype=np.uint8)
        m[rng.integers(10, 30):rng.integers(31, 50), 10:50] = 255
        features.append(extract_image_features(m, fov_mask))
        labels.append(0)
    for _ in range(10):
        m = np.zeros((64, 64), dtype=np.uint8)
        for _ in range(5):
            r = rng.integers(5, 55)
            m[r:r+5, 5:60] = 255
        features.append(extract_image_features(m, fov_mask))
        labels.append(1)

    X = np.vstack(features)
    y = np.array(labels)
    model, scaler = train(X, y, method="random_forest")

    result = predict(model, scaler, vessel_mask, fov_mask)
    assert "label" in result
    assert result["label"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
    assert "features" in result
