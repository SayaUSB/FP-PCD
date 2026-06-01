import numpy as np
import cv2
from skimage.filters import frangi
from skimage.feature import local_binary_pattern
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def extract_features(
    img_rgb: np.ndarray,
    vessel_enhanced: np.ndarray,
    fov_mask: np.ndarray,
) -> np.ndarray:
    green = img_rgb[:, :, 1].astype(np.float32) / 255.0
    frangi_resp = frangi(green).astype(np.float32)
    gx = cv2.Sobel(green, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(green, cv2.CV_32F, 0, 1)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    lbp = local_binary_pattern(img_rgb[:, :, 1], P=8, R=1, method="uniform").astype(np.float32)
    lbp_max = lbp.max()
    lbp_norm = lbp / lbp_max if lbp_max > 0 else lbp
    ve_norm = cv2.normalize(vessel_enhanced, None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)

    mask = fov_mask > 0
    features = np.column_stack([
        green[mask],
        ve_norm[mask],
        frangi_resp[mask],
        grad_mag[mask],
        lbp_norm[mask],
    ]).astype(np.float32)
    return features


def train(
    features: np.ndarray,
    labels: np.ndarray,
    method: str = "svm",
) -> object:
    sm = SMOTE(random_state=42)
    X, y = sm.fit_resample(features, labels)
    if method == "svm":
        model = SVC(kernel="rbf", probability=True, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model


def predict(
    model: object,
    img_rgb: np.ndarray,
    vessel_enhanced: np.ndarray,
    fov_mask: np.ndarray,
) -> np.ndarray:
    features = extract_features(img_rgb, vessel_enhanced, fov_mask)
    preds = model.predict(features)
    result = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    result[fov_mask > 0] = (preds > 0).astype(np.uint8) * 255
    return result
