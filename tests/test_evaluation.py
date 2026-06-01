import numpy as np
import pytest


@pytest.fixture
def perfect_pred():
    """4x4 images: vessel = top-left 2x2."""
    gt = np.zeros((4, 4), dtype=np.uint8)
    gt[:2, :2] = 255
    pred = gt.copy()
    fov = np.full((4, 4), 255, dtype=np.uint8)
    return pred, gt, fov


@pytest.fixture
def imperfect_pred():
    """GT has vessel at top-left; pred misses one pixel, adds one FP."""
    gt = np.zeros((4, 4), dtype=np.uint8)
    gt[:2, :2] = 255                        # 4 vessel pixels
    pred = np.zeros((4, 4), dtype=np.uint8)
    pred[:2, :2] = 255
    pred[0, 0] = 0                           # FN: missed one vessel pixel
    pred[3, 3] = 255                         # FP: wrong pixel
    fov = np.full((4, 4), 255, dtype=np.uint8)
    return pred, gt, fov


def test_evaluate_perfect_prediction(perfect_pred):
    from retina_seg.evaluation.metrics import evaluate
    pred, gt, fov = perfect_pred
    result = evaluate(pred, gt, fov)
    assert result["SE"] == pytest.approx(1.0)
    assert result["SP"] == pytest.approx(1.0)
    assert result["ACC"] == pytest.approx(1.0)


def test_evaluate_imperfect_prediction(imperfect_pred):
    from retina_seg.evaluation.metrics import evaluate
    pred, gt, fov = imperfect_pred
    result = evaluate(pred, gt, fov)
    assert 0.0 < result["SE"] < 1.0
    assert 0.0 < result["SP"] < 1.0
    assert "TP" in result and "FP" in result and "FN" in result and "TN" in result


def test_make_overlay_shape_and_dtype():
    from retina_seg.evaluation.metrics import make_overlay
    img_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    img_rgb[:] = 128
    pred = np.zeros((64, 64), dtype=np.uint8)
    pred[10:20, 10:50] = 255
    gt = np.zeros((64, 64), dtype=np.uint8)
    gt[10:20, 10:45] = 255
    fov = np.full((64, 64), 255, dtype=np.uint8)
    overlay = make_overlay(img_rgb, pred, gt, fov)
    assert overlay.shape == (64, 64, 3)
    assert overlay.dtype == np.uint8


def test_compute_auc_returns_float():
    from retina_seg.evaluation.metrics import compute_auc
    rng = np.random.default_rng(42)
    enhanced = rng.random((64, 64)).astype(np.float32)
    gt = np.zeros((64, 64), dtype=np.uint8)
    gt[20:30, 20:40] = 255
    fov = np.full((64, 64), 255, dtype=np.uint8)
    auc_val, fpr, tpr = compute_auc(enhanced, gt, fov)
    assert 0.0 <= auc_val <= 1.0
    assert len(fpr) == len(tpr)
