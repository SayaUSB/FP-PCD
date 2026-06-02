import numpy as np
import cv2
from skimage.morphology import skeletonize
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib


# ── Feature extraction (image-level) ────────────────────────────────────────

def _vessel_density(vessel_mask: np.ndarray, fov_mask: np.ndarray) -> float:
    fov_area = (fov_mask > 0).sum()
    if fov_area == 0:
        return 0.0
    return float((vessel_mask > 0).sum()) / fov_area


def _mean_vessel_width(vessel_mask: np.ndarray) -> float:
    binary = (vessel_mask > 0).astype(np.uint8)
    skeleton = skeletonize(binary).astype(np.uint8)
    skel_pixels = skeleton.sum()
    if skel_pixels == 0:
        return 0.0
    vessel_area = binary.sum()
    return float(vessel_area) / skel_pixels


def _tortuosity(vessel_mask: np.ndarray) -> float:
    """
    Ratio of vessel arc length to chord length, averaged over all connected segments.
    Approximated via skeleton: higher = more curved vessels.
    """
    from skimage.measure import label as sk_label
    skeleton = skeletonize((vessel_mask > 0).astype(np.uint8))
    labeled = sk_label(skeleton)
    n_labels = labeled.max()
    if n_labels == 0:
        return 0.0

    tortuosity_vals = []
    for lbl in range(1, n_labels + 1):
        pts = np.argwhere(labeled == lbl)
        if len(pts) < 3:
            continue
        arc_length = len(pts)
        chord = np.linalg.norm(pts[0] - pts[-1])
        if chord > 0:
            tortuosity_vals.append(arc_length / chord)

    return float(np.mean(tortuosity_vals)) if tortuosity_vals else 0.0


def _branching_points(vessel_mask: np.ndarray) -> float:
    """Count branching points in the vessel skeleton, normalized by FOV area."""
    skeleton = skeletonize((vessel_mask > 0).astype(np.uint8)).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton, -1, kernel) * skeleton
    branch_pts = (neighbor_count >= 4).sum()
    total = skeleton.sum()
    return float(branch_pts) / total if total > 0 else 0.0


def _fractal_dimension(vessel_mask: np.ndarray) -> float:
    """Box-counting fractal dimension of the vessel mask."""
    binary = (vessel_mask > 0).astype(np.uint8)
    min_dim = min(binary.shape)
    sizes = [2, 4, 8, 16, 32]
    sizes = [s for s in sizes if s < min_dim // 2]
    if len(sizes) < 2:
        return 0.0

    counts = []
    for size in sizes:
        h, w = binary.shape
        count = 0
        for i in range(0, h, size):
            for j in range(0, w, size):
                if binary[i:i+size, j:j+size].any():
                    count += 1
        counts.append(count)

    log_sizes = np.log(1.0 / np.array(sizes, dtype=np.float64))
    log_counts = np.log(np.array(counts, dtype=np.float64) + 1e-9)
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    return float(coeffs[0])


def extract_image_features(vessel_mask: np.ndarray, fov_mask: np.ndarray) -> np.ndarray:
    """
    Extract 5 vessel-level features from a segmented vessel mask.
    Returns shape (5,) float32 vector.
    """
    density = _vessel_density(vessel_mask, fov_mask)
    width = _mean_vessel_width(vessel_mask)
    tortuosity = _tortuosity(vessel_mask)
    branching = _branching_points(vessel_mask)
    fractal = _fractal_dimension(vessel_mask)
    return np.array([density, width, tortuosity, branching, fractal], dtype=np.float32)


FEATURE_NAMES = [
    "vessel_density",
    "mean_vessel_width",
    "tortuosity",
    "branching_points",
    "fractal_dimension",
]


# ── Training ─────────────────────────────────────────────────────────────────

def train(
    features: np.ndarray,
    labels: np.ndarray,
    method: str = "random_forest",
) -> tuple[object, StandardScaler]:
    """
    Train image-level classifier (normal vs abnormal).
    Returns (model, scaler) — both needed for predict().
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    y = labels

    # SMOTE only if both classes present and enough samples
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) == 2 and counts.min() >= 2:
        k = min(5, counts.min() - 1)
        sm = SMOTE(random_state=42, k_neighbors=k)
        X, y = sm.fit_resample(X, y)

    if method == "svm":
        model = SVC(kernel="rbf", probability=True, random_state=42, C=10, gamma="scale")
    else:
        model = RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
        )
    model.fit(X, y)
    return model, scaler


# ── Prediction ───────────────────────────────────────────────────────────────

def predict(
    model: object,
    scaler: StandardScaler,
    vessel_mask: np.ndarray,
    fov_mask: np.ndarray,
) -> dict:
    """
    Predict anomaly label for a single image.
    Returns dict with label (0/1), probability, and feature values.
    """
    feats = extract_image_features(vessel_mask, fov_mask).reshape(1, -1)
    feats_scaled = scaler.transform(feats)
    label = int(model.predict(feats_scaled)[0])
    prob = float(model.predict_proba(feats_scaled)[0][1])
    return {
        "label": label,
        "label_str": "Abnormal (DR)" if label == 1 else "Normal",
        "probability": prob,
        "features": dict(zip(FEATURE_NAMES, feats[0].tolist())),
    }


# ── Persistence ──────────────────────────────────────────────────────────────

def save_model(model: object, scaler: StandardScaler, path: str) -> None:
    joblib.dump({"model": model, "scaler": scaler}, path)


def load_model(path: str) -> tuple[object, StandardScaler]:
    data = joblib.load(path)
    return data["model"], data["scaler"]
