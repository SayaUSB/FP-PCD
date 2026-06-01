import numpy as np
import cv2
from sklearn.metrics import roc_curve, auc as sklearn_auc


def evaluate(
    pred: np.ndarray,
    gt: np.ndarray,
    fov_mask: np.ndarray,
) -> dict:
    mask = fov_mask > 0
    pred_b = (pred[mask] > 0).astype(int)
    gt_b = (gt[mask] > 0).astype(int)

    TP = int(np.sum((pred_b == 1) & (gt_b == 1)))
    TN = int(np.sum((pred_b == 0) & (gt_b == 0)))
    FP = int(np.sum((pred_b == 1) & (gt_b == 0)))
    FN = int(np.sum((pred_b == 0) & (gt_b == 1)))
    total = TP + TN + FP + FN

    SE = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    SP = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    ACC = (TP + TN) / total if total > 0 else 0.0

    return {"SE": SE, "SP": SP, "ACC": ACC, "TP": TP, "TN": TN, "FP": FP, "FN": FN}


def compute_auc(
    vessel_enhanced: np.ndarray,
    gt: np.ndarray,
    fov_mask: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    mask = fov_mask > 0
    scores = cv2.normalize(vessel_enhanced, None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)
    scores_flat = scores[mask].ravel()
    labels_flat = (gt[mask] > 0).astype(int).ravel()
    fpr, tpr, _ = roc_curve(labels_flat, scores_flat)
    return float(sklearn_auc(fpr, tpr)), fpr, tpr


def make_overlay(
    img_rgb: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    fov_mask: np.ndarray,
) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    m = fov_mask > 0
    pred_b = (pred > 0) & m
    gt_b = (gt > 0) & m
    overlay[pred_b & gt_b] = [0, 255, 0]     # TP: green
    overlay[pred_b & ~gt_b] = [255, 0, 0]    # FP: red
    overlay[~pred_b & gt_b] = [0, 0, 255]    # FN: blue
    return overlay
