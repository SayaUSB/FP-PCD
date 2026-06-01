# Retinal Vessel Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete retinal blood vessel segmentation pipeline (CLAHE + Matched Filter) with a Streamlit demo app, evaluated on DRIVE/STARE/CHASE_DB1 datasets.

**Architecture:** Modular Python package under `src/retina_seg/` — one subpackage per pipeline stage, each exposing a `run()` function with numpy-in/numpy-out interface. `app.py` (Streamlit) chains all stages. ML pipeline is decoupled and optional.

**Tech Stack:** Python 3.10+, OpenCV (`opencv-python`), scikit-image, scikit-learn, imbalanced-learn (SMOTE), scipy, numpy, matplotlib, streamlit, pytest, Pillow, pandas

---

## File Map

```
retina-vessel-seg/
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── README.md
├── app.py
├── src/
│   └── retina_seg/
│       ├── __init__.py
│       ├── dataset/
│       │   ├── __init__.py
│       │   ├── loader.py       # load_drive(), load_stare(), load_chase()
│       │   └── utils.py        # read_image(), read_mask(), list_files()
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   └── pipeline.py     # run(img_rgb, fov_mask) -> (corrected, fov_mask)
│       ├── enhancement/
│       │   ├── __init__.py
│       │   └── pipeline.py     # run(img, return_intermediates) -> enhanced
│       ├── segmentation/
│       │   ├── __init__.py
│       │   └── pipeline.py     # run(img, fov_mask) -> binary uint8
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── metrics.py      # evaluate(), compute_auc(), make_overlay()
│       └── ml/
│           ├── __init__.py
│           └── pipeline.py     # extract_features(), train(), predict()
├── tests/
│   ├── conftest.py
│   ├── test_utils.py
│   ├── test_loader.py
│   ├── test_preprocessing.py
│   ├── test_enhancement.py
│   ├── test_segmentation.py
│   ├── test_evaluation.py
│   └── test_ml.py
└── data/
    ├── DRIVE/README.md
    ├── STARE/README.md
    └── CHASE_DB1/README.md
```

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: all `src/retina_seg/**/__init__.py` files

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "retina-seg"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "opencv-python",
    "scikit-image",
    "scikit-learn",
    "imbalanced-learn",
    "numpy",
    "matplotlib",
    "streamlit",
    "joblib",
    "Pillow",
    "scipy",
    "pandas",
]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Create `requirements.txt`**

```
opencv-python
scikit-image
scikit-learn
imbalanced-learn
numpy
matplotlib
streamlit
joblib
Pillow
scipy
pandas
pytest
```

- [ ] **Step 3: Create `.gitignore`**

```
__pycache__/
*.pyc
*.pyo
.eggs/
*.egg-info/
dist/
build/
.env
*.pkl
*.joblib
data/DRIVE/
data/STARE/
data/CHASE_DB1/
.streamlit/
```

- [ ] **Step 4: Create package directory structure**

```bash
mkdir -p src/retina_seg/{dataset,preprocessing,enhancement,segmentation,evaluation,ml}
mkdir -p tests data/DRIVE data/STARE data/CHASE_DB1
touch src/retina_seg/__init__.py
touch src/retina_seg/{dataset,preprocessing,enhancement,segmentation,evaluation,ml}/__init__.py
```

- [ ] **Step 5: Install in editable mode**

```bash
pip install -e .
```

Expected: `Successfully installed retina-seg-0.1.0`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml requirements.txt .gitignore src/
git commit -m "feat: project scaffold and package setup"
```

---

## Task 2: Dataset Utilities

**Files:**
- Create: `src/retina_seg/dataset/utils.py`
- Create: `tests/conftest.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1: Write failing tests**

Create `tests/conftest.py`:
```python
import numpy as np
import pytest

@pytest.fixture
def synthetic_rgb():
    """256x256 RGB fundus-like image: bright circle on black."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    import cv2
    cv2.circle(img, (128, 128), 100, (180, 150, 120), -1)
    # Add a dark horizontal vessel
    img[120:125, 30:220] = (100, 80, 60)
    return img

@pytest.fixture
def synthetic_mask():
    """256x256 binary FOV mask."""
    import numpy as np
    mask = np.zeros((256, 256), dtype=np.uint8)
    import cv2
    cv2.circle(mask, (128, 128), 100, 255, -1)
    return mask

@pytest.fixture
def synthetic_gt():
    """256x256 ground truth: horizontal vessel at rows 120-125."""
    gt = np.zeros((256, 256), dtype=np.uint8)
    gt[120:125, 30:220] = 255
    return gt
```

Create `tests/test_utils.py`:
```python
import numpy as np
import pytest
import tempfile
import os
from PIL import Image

def test_read_image_returns_rgb_uint8(tmp_path):
    from retina_seg.dataset.utils import read_image
    img_path = str(tmp_path / "test.png")
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[10:20, 10:20] = [255, 0, 0]
    Image.fromarray(arr).save(img_path)
    result = read_image(img_path)
    assert result.shape == (64, 64, 3)
    assert result.dtype == np.uint8

def test_read_mask_returns_binary(tmp_path):
    from retina_seg.dataset.utils import read_mask
    mask_path = str(tmp_path / "mask.png")
    arr = np.zeros((64, 64), dtype=np.uint8)
    arr[10:50, 10:50] = 255
    Image.fromarray(arr).save(mask_path)
    result = read_mask(mask_path)
    assert result.shape == (64, 64)
    assert set(np.unique(result)).issubset({0, 255})

def test_list_files_sorted(tmp_path):
    from retina_seg.dataset.utils import list_files
    for name in ["03.png", "01.png", "02.png"]:
        (tmp_path / name).write_bytes(b"")
    result = list_files(str(tmp_path), ".png")
    assert result == sorted(result)
    assert len(result) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_utils.py -v
```

Expected: `ModuleNotFoundError: No module named 'retina_seg.dataset.utils'`

- [ ] **Step 3: Implement `src/retina_seg/dataset/utils.py`**

```python
import os
import numpy as np
import cv2


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    return (mask > 127).astype(np.uint8) * 255


def list_files(folder: str, ext: str) -> list[str]:
    ext = ext.lower()
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(ext)
    ]
    return sorted(files)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_utils.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/retina_seg/dataset/utils.py tests/
git commit -m "feat: dataset utilities (read_image, read_mask, list_files)"
```

---

## Task 3: Dataset Loaders

**Files:**
- Create: `src/retina_seg/dataset/loader.py`
- Create: `tests/test_loader.py`

Each loader returns a `list[dict]` where each dict has:
`{"image": np.ndarray (H,W,3 uint8), "mask": np.ndarray (H,W uint8), "gt1": np.ndarray (H,W uint8), "gt2": np.ndarray|None, "name": str}`

- [ ] **Step 1: Write failing tests**

Create `tests/test_loader.py`:
```python
import numpy as np
import pytest
import os
from PIL import Image


def _make_fake_image(path, size=(64, 64), mode="RGB"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.full((*size, 3) if mode == "RGB" else size, 128, dtype=np.uint8)
    img = Image.fromarray(arr if mode == "RGB" else arr, mode=mode)
    img.save(path)


def test_load_drive_returns_list_of_dicts(tmp_path):
    from retina_seg.dataset.loader import load_drive
    # Build fake DRIVE structure
    for split in ["training", "test"]:
        for i in range(1, 3):
            name = f"0{i}_{split}"
            _make_fake_image(str(tmp_path / split / "images" / f"{name}.png"))
            _make_fake_image(str(tmp_path / split / "mask" / f"{name}_mask.png"), mode="L")
            _make_fake_image(str(tmp_path / split / "1st_manual" / f"{name}_manual1.png"), mode="L")
    
    samples = load_drive(str(tmp_path), split="training")
    assert len(samples) == 2
    assert "image" in samples[0]
    assert "mask" in samples[0]
    assert "gt1" in samples[0]
    assert samples[0]["image"].shape[2] == 3
    assert samples[0]["mask"].ndim == 2


def test_load_stare_returns_list_of_dicts(tmp_path):
    from retina_seg.dataset.loader import load_stare
    imgs_dir = tmp_path / "images"
    ah_dir = tmp_path / "labels-ah"
    imgs_dir.mkdir(); ah_dir.mkdir()
    for i in range(1, 3):
        _make_fake_image(str(imgs_dir / f"im000{i}.png"))
        _make_fake_image(str(ah_dir / f"im000{i}.ah.png"), mode="L")
    
    samples = load_stare(str(tmp_path))
    assert len(samples) == 2
    assert samples[0]["image"].shape[2] == 3


def test_load_chase_returns_list_of_dicts(tmp_path):
    from retina_seg.dataset.loader import load_chase
    for i in range(1, 3):
        _make_fake_image(str(tmp_path / f"Image_0{i}L.jpg"))
        _make_fake_image(str(tmp_path / f"Image_0{i}L_1stHO.png"), mode="L")
        _make_fake_image(str(tmp_path / f"Image_0{i}L_2ndHO.png"), mode="L")
    
    samples = load_chase(str(tmp_path))
    assert len(samples) == 2
    assert "gt2" in samples[0]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_loader.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement `src/retina_seg/dataset/loader.py`**

```python
import os
import glob
import numpy as np
from .utils import read_image, read_mask, list_files


def load_drive(root: str, split: str = "training") -> list[dict]:
    """
    DRIVE structure:
      <root>/<split>/images/*.tif (or .png)
      <root>/<split>/mask/*_mask.*
      <root>/<split>/1st_manual/*_manual1.*
      <root>/<split>/2nd_manual/*_manual2.*  (test split only)
    """
    split_dir = os.path.join(root, split)
    img_files = sorted(glob.glob(os.path.join(split_dir, "images", "*")))
    mask_files = sorted(glob.glob(os.path.join(split_dir, "mask", "*")))
    gt1_files = sorted(glob.glob(os.path.join(split_dir, "1st_manual", "*")))
    gt2_dir = os.path.join(split_dir, "2nd_manual")
    gt2_files = sorted(glob.glob(os.path.join(gt2_dir, "*"))) if os.path.isdir(gt2_dir) else []

    samples = []
    for i, img_path in enumerate(img_files):
        name = os.path.splitext(os.path.basename(img_path))[0]
        sample = {
            "name": name,
            "image": read_image(img_path),
            "mask": read_mask(mask_files[i]) if i < len(mask_files) else None,
            "gt1": read_mask(gt1_files[i]) if i < len(gt1_files) else None,
            "gt2": read_mask(gt2_files[i]) if i < len(gt2_files) else None,
        }
        samples.append(sample)
    return samples


def load_stare(root: str) -> list[dict]:
    """
    STARE structure:
      <root>/images/*.ppm (or other format)
      <root>/labels-ah/*.ah.ppm (first annotator)
      <root>/labels-vk/*.vk.ppm (second annotator, optional)
    """
    img_files = sorted(glob.glob(os.path.join(root, "images", "*")))
    ah_files = sorted(glob.glob(os.path.join(root, "labels-ah", "*")))
    vk_dir = os.path.join(root, "labels-vk")
    vk_files = sorted(glob.glob(os.path.join(vk_dir, "*"))) if os.path.isdir(vk_dir) else []

    samples = []
    for i, img_path in enumerate(img_files):
        name = os.path.splitext(os.path.basename(img_path))[0]
        h, w = read_image(img_path).shape[:2]
        sample = {
            "name": name,
            "image": read_image(img_path),
            "mask": _generate_circular_mask(h, w),
            "gt1": read_mask(ah_files[i]) if i < len(ah_files) else None,
            "gt2": read_mask(vk_files[i]) if i < len(vk_files) else None,
        }
        samples.append(sample)
    return samples


def load_chase(root: str) -> list[dict]:
    """
    CHASE_DB1 structure (all files in one flat directory):
      Image_01L.jpg, Image_01L_1stHO.png, Image_01L_2ndHO.png
      Image_01R.jpg, Image_01R_1stHO.png, Image_01R_2ndHO.png ...
    """
    img_files = sorted([
        f for f in glob.glob(os.path.join(root, "*.jpg"))
        if "_HO" not in f
    ])

    samples = []
    for img_path in img_files:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        gt1_path = os.path.join(root, f"{stem}_1stHO.png")
        gt2_path = os.path.join(root, f"{stem}_2ndHO.png")
        img = read_image(img_path)
        h, w = img.shape[:2]
        sample = {
            "name": stem,
            "image": img,
            "mask": _generate_circular_mask(h, w),
            "gt1": read_mask(gt1_path) if os.path.isfile(gt1_path) else None,
            "gt2": read_mask(gt2_path) if os.path.isfile(gt2_path) else None,
        }
        samples.append(sample)
    return samples


def _generate_circular_mask(h: int, w: int, margin_ratio: float = 0.03) -> np.ndarray:
    """Generate a circular FOV mask for datasets without pre-supplied masks."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) / 2 * (1 - margin_ratio))
    import cv2
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_loader.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/retina_seg/dataset/loader.py tests/test_loader.py
git commit -m "feat: dataset loaders for DRIVE, STARE, CHASE_DB1"
```

---

## Task 4: Preprocessing Pipeline

**Files:**
- Create: `src/retina_seg/preprocessing/pipeline.py`
- Create: `tests/test_preprocessing.py`

`run(img_rgb, fov_mask=None) -> tuple[np.ndarray, np.ndarray]`
Returns: `(corrected_uint8, fov_mask_uint8)` where corrected is grayscale.

- [ ] **Step 1: Write failing tests**

Create `tests/test_preprocessing.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_preprocessing.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement `src/retina_seg/preprocessing/pipeline.py`**

```python
import numpy as np
import cv2


def extract_green(img_rgb: np.ndarray) -> np.ndarray:
    return img_rgb[:, :, 1]


def generate_fov_mask(img_rgb: np.ndarray, threshold: int = 20) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
    return mask


def correct_illumination(img_green: np.ndarray, radius: int = 70) -> np.ndarray:
    diam = 2 * radius + 1
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diam, diam))
    background = cv2.morphologyEx(img_green, cv2.MORPH_OPEN, disk)
    # Subtract: normalize to [0,255] to keep vessels relatively dark
    diff = img_green.astype(np.float32) - background.astype(np.float32)
    corrected = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return corrected.astype(np.uint8)


def run(
    img_rgb: np.ndarray,
    fov_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    green = extract_green(img_rgb)
    if fov_mask is None:
        fov_mask = generate_fov_mask(img_rgb)
    corrected = correct_illumination(green)
    # Set outside-FOV pixels to mean inside FOV to avoid border effects
    inside = corrected[fov_mask > 0]
    fill_val = int(inside.mean()) if inside.size > 0 else 0
    result = corrected.copy()
    result[fov_mask == 0] = fill_val
    return result, fov_mask
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_preprocessing.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/retina_seg/preprocessing/pipeline.py tests/test_preprocessing.py
git commit -m "feat: preprocessing pipeline (green channel, FOV mask, illumination correction)"
```

---

## Task 5: Enhancement Pipeline

**Files:**
- Create: `src/retina_seg/enhancement/pipeline.py`
- Create: `tests/test_enhancement.py`

`run(img, return_intermediates=False) -> np.ndarray | tuple[np.ndarray, dict]`
Returns float32 vessel-enhanced image (higher values = likely vessel).

- [ ] **Step 1: Write failing tests**

Create `tests/test_enhancement.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_enhancement.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement `src/retina_seg/enhancement/pipeline.py`**

```python
import numpy as np
import cv2
from scipy.ndimage import rotate as ndimage_rotate


def _build_mf_kernel(sigma: float = 2.0, L: int = 9) -> np.ndarray:
    """
    Square matched filter kernel for a horizontal dark vessel.
    Cross-section profile: -exp(-row_offset²/2σ²), active for |col_offset| ≤ L//2.
    Mean-subtracted so uniform backgrounds give zero response.
    """
    half = max(int(np.ceil(3 * sigma)), L // 2)
    size = 2 * half + 1
    center = half
    rows, cols = np.mgrid[0:size, 0:size]
    row_off = (rows - center).astype(np.float64)
    col_off = (cols - center).astype(np.float64)
    kernel = np.where(
        np.abs(col_off) <= L // 2,
        -np.exp(-row_off ** 2 / (2.0 * sigma ** 2)),
        0.0,
    )
    kernel -= kernel.mean()
    return kernel.astype(np.float32)


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(img)


def apply_gaussian(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    ksize = int(6 * sigma + 1) | 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def apply_matched_filter(
    img: np.ndarray,
    sigma: float = 2.0,
    L: int = 9,
    n_angles: int = 12,
) -> np.ndarray:
    base = _build_mf_kernel(sigma, L)
    img_f = img.astype(np.float32)
    response = np.full_like(img_f, -np.inf)
    for k in range(n_angles):
        angle = k * 180.0 / n_angles
        rotated = ndimage_rotate(base, angle, reshape=False, order=1).astype(np.float32)
        filtered = cv2.filter2D(img_f, -1, rotated)
        response = np.maximum(response, filtered)
    return response


def run(
    img: np.ndarray,
    return_intermediates: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    clahe_out = apply_clahe(img)
    gauss_out = apply_gaussian(clahe_out)
    enhanced = apply_matched_filter(gauss_out)
    if return_intermediates:
        return enhanced, {"clahe": clahe_out, "gaussian": gauss_out}
    return enhanced
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_enhancement.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/retina_seg/enhancement/pipeline.py tests/test_enhancement.py
git commit -m "feat: enhancement pipeline (CLAHE, Gaussian, Matched Filter 12 orientations)"
```

---

## Task 6: Segmentation Pipeline

**Files:**
- Create: `src/retina_seg/segmentation/pipeline.py`
- Create: `tests/test_segmentation.py`

`run(img, fov_mask, return_intermediates=False) -> np.ndarray (binary uint8)`

- [ ] **Step 1: Write failing tests**

Create `tests/test_segmentation.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_segmentation.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement `src/retina_seg/segmentation/pipeline.py`**

```python
import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage.morphology import (
    remove_small_objects,
    binary_opening,
    binary_closing,
    disk,
)


def apply_otsu(img: np.ndarray, fov_mask: np.ndarray) -> np.ndarray:
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    roi_pixels = img_norm[fov_mask > 0]
    thresh = threshold_otsu(roi_pixels)
    binary = (img_norm >= thresh).astype(np.uint8) * 255
    binary[fov_mask == 0] = 0
    return binary


def morphological_postprocess(binary: np.ndarray, min_area: int = 50) -> np.ndarray:
    bool_img = binary > 0
    selem = disk(1)
    opened = binary_opening(bool_img, selem)
    closed = binary_closing(opened, selem)
    cleaned = remove_small_objects(closed, min_size=min_area)
    return (cleaned * 255).astype(np.uint8)


def run(
    img: np.ndarray,
    fov_mask: np.ndarray,
    return_intermediates: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    binary = apply_otsu(img, fov_mask)
    result = morphological_postprocess(binary)
    result[fov_mask == 0] = 0
    if return_intermediates:
        return result, {"otsu": binary}
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_segmentation.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/retina_seg/segmentation/pipeline.py tests/test_segmentation.py
git commit -m "feat: segmentation pipeline (Otsu thresholding + morphological post-processing)"
```

---

## Task 7: Evaluation Metrics

**Files:**
- Create: `src/retina_seg/evaluation/metrics.py`
- Create: `tests/test_evaluation.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_evaluation.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_evaluation.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement `src/retina_seg/evaluation/metrics.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_evaluation.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/retina_seg/evaluation/metrics.py tests/test_evaluation.py
git commit -m "feat: evaluation metrics (SE, SP, ACC, AUC, overlay visualization)"
```

---

## Task 8: ML Pipeline (Optional)

**Files:**
- Create: `src/retina_seg/ml/pipeline.py`
- Create: `tests/test_ml.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ml.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ml.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement `src/retina_seg/ml/pipeline.py`**

```python
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
    """Returns (N_fov_pixels, 5) float32 feature matrix."""
    green = img_rgb[:, :, 1].astype(np.float32) / 255.0
    frangi_resp = frangi(green).astype(np.float32)
    gx = cv2.Sobel(green, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(green, cv2.CV_32F, 0, 1)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    lbp = local_binary_pattern(green, P=8, R=1, method="uniform").astype(np.float32)
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ml.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/retina_seg/ml/pipeline.py tests/test_ml.py
git commit -m "feat: ML pipeline (feature extraction, SMOTE, SVM/RF)"
```

---

## Task 9: Streamlit Demo App

**Files:**
- Create: `app.py`

- [ ] **Step 1: Implement `app.py`**

```python
import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from retina_seg.preprocessing.pipeline import run as preprocess
from retina_seg.enhancement.pipeline import run as enhance
from retina_seg.segmentation.pipeline import run as segment
from retina_seg.evaluation.metrics import evaluate, compute_auc, make_overlay
from retina_seg.dataset.loader import load_drive, load_stare, load_chase

st.set_page_config(page_title="Retinal Vessel Segmentation", layout="wide")
st.title("Segmentasi Pembuluh Darah Retina — CLAHE + Matched Filter")

tab1, tab2 = st.tabs(["Single Image", "Batch Evaluation"])

# ── Tab 1: Single Image ──────────────────────────────────────────────────────
with tab1:
    st.header("Pipeline on Single Image")
    uploaded = st.file_uploader(
        "Upload citra fundus (JPG/PNG/TIFF)",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
    )

    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        img_rgb = np.array(img_pil)
        st.image(img_rgb, caption="Input Image", use_column_width=False, width=300)

        if st.button("Run Pipeline"):
            with st.spinner("Processing..."):
                corrected, fov_mask = preprocess(img_rgb)
                enhanced, inter = enhance(corrected, return_intermediates=True)
                segmented, seg_inter = segment(enhanced, fov_mask, return_intermediates=True)

            st.subheader("Pipeline Stages")
            cols = st.columns(4)
            cols[0].image(corrected, caption="1. Green + Illum. Corrected", clamp=True)
            cols[1].image(inter["clahe"], caption="2. CLAHE")
            cols[2].image(
                cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                caption="3. Matched Filter Response",
            )
            cols[3].image(segmented, caption="4. Segmentation Output")

            buf = io.BytesIO()
            Image.fromarray(segmented).save(buf, format="PNG")
            st.download_button(
                "Download Segmentation",
                buf.getvalue(),
                "segmentation.png",
                "image/png",
            )

# ── Tab 2: Batch Evaluation ──────────────────────────────────────────────────
with tab2:
    st.header("Batch Evaluation on Dataset")
    dataset_name = st.selectbox("Dataset", ["DRIVE", "STARE", "CHASE_DB1"])
    data_path = st.text_input(
        "Path ke folder dataset",
        placeholder="/home/user/data/DRIVE",
    )
    split = st.radio("Split (DRIVE only)", ["training", "test"], horizontal=True)

    if st.button("Run Batch Evaluation") and data_path:
        loader_map = {
            "DRIVE": lambda p: load_drive(p, split=split),
            "STARE": load_stare,
            "CHASE_DB1": load_chase,
        }
        try:
            samples = loader_map[dataset_name](data_path)
        except Exception as exc:
            st.error(f"Gagal load dataset: {exc}")
            samples = []

        if samples:
            rows = []
            progress = st.progress(0)
            for i, sample in enumerate(samples):
                corrected, fov_mask = preprocess(sample["image"], sample.get("mask"))
                enhanced = enhance(corrected)
                segmented = segment(enhanced, fov_mask)
                gt = sample["gt1"]
                if gt is None:
                    continue
                metrics = evaluate(segmented, gt, fov_mask)
                auc_val, _, _ = compute_auc(enhanced, gt, fov_mask)
                metrics["AUC"] = round(auc_val, 4)
                metrics["name"] = sample["name"]
                rows.append(metrics)
                progress.progress((i + 1) / len(samples))

            if rows:
                df = pd.DataFrame(rows).set_index("name")
                st.subheader("Per-Image Results")
                st.dataframe(df[["SE", "SP", "ACC", "AUC"]].round(4))
                st.subheader("Aggregate Statistics")
                st.dataframe(df[["SE", "SP", "ACC", "AUC"]].describe().round(4))

                st.subheader("Contoh Overlay (citra pertama)")
                s = samples[0]
                corrected, fov_mask = preprocess(s["image"], s.get("mask"))
                enhanced = enhance(corrected)
                segmented = segment(enhanced, fov_mask)
                overlay = make_overlay(s["image"], segmented, s["gt1"], fov_mask)
                st.image(overlay, caption="Hijau=TP, Merah=FP, Biru=FN", width=400)
```

- [ ] **Step 2: Test the app starts without error**

```bash
streamlit run app.py --server.headless true &
sleep 3
curl -s http://localhost:8501 | grep -q "Streamlit" && echo "OK" || echo "FAIL"
pkill -f "streamlit run"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: Streamlit demo app with single-image pipeline and batch evaluation"
```

---

## Task 10: Dataset READMEs and Project README

**Files:**
- Create: `data/DRIVE/README.md`
- Create: `data/STARE/README.md`
- Create: `data/CHASE_DB1/README.md`
- Create: `README.md`

- [ ] **Step 1: Create `data/DRIVE/README.md`**

```markdown
# DRIVE Dataset

Download from: https://drive.grand-challenge.org  
(Requires free registration)

After downloading, extract so the directory looks like:

```
data/DRIVE/
├── training/
│   ├── images/       # 01_training.tif ... 20_training.tif
│   ├── mask/         # 01_training_mask.gif ... 20_training_mask.gif
│   └── 1st_manual/   # 01_manual1.gif ... 20_manual1.gif
└── test/
    ├── images/       # 01_test.tif ... 20_test.tif
    ├── mask/         # 01_test_mask.gif ... 20_test_mask.gif
    ├── 1st_manual/   # 01_manual1.gif ... 20_manual1.gif
    └── 2nd_manual/   # 01_manual2.gif ... 20_manual2.gif
```
```

- [ ] **Step 2: Create `data/STARE/README.md`**

```markdown
# STARE Dataset

Download from: https://cecas.clemson.edu/~ahoover/stare  
(Direct download, no registration needed)

Files needed:
- `stare-images.tar` → extract to `data/STARE/images/`
- `labels-ah.tar` → extract to `data/STARE/labels-ah/`
- `labels-vk.tar` → extract to `data/STARE/labels-vk/`

Expected structure:
```
data/STARE/
├── images/      # im0001.ppm ... im0020.ppm
├── labels-ah/   # im0001.ah.ppm ... (first annotator)
└── labels-vk/   # im0001.vk.ppm ... (second annotator)
```
```

- [ ] **Step 3: Create `data/CHASE_DB1/README.md`**

```markdown
# CHASE_DB1 Dataset

Download from: https://www.kaggle.com/datasets/khoongweihao/chasedb1  
(Requires free Kaggle account)

After downloading, place all files flat in `data/CHASE_DB1/`:

```
data/CHASE_DB1/
├── Image_01L.jpg
├── Image_01L_1stHO.png
├── Image_01L_2ndHO.png
├── Image_01R.jpg
...
```
```

- [ ] **Step 4: Create `README.md`**

```markdown
# Retinal Vessel Segmentation — CLAHE + Matched Filter

Segmentasi pembuluh darah retina menggunakan pipeline image processing murni (tanpa deep learning).

## Pipeline

```
RGB Image → Green Channel → FOV Mask → Illumination Correction
         → CLAHE → Gaussian → Matched Filter (12 orientations)
         → Otsu Thresholding → Morphological Post-processing
         → Binary Segmentation
```

## Installation

```bash
pip install -e .
```

## Dataset Setup

Lihat instruksi download di:
- `data/DRIVE/README.md`
- `data/STARE/README.md`
- `data/CHASE_DB1/README.md`

## Usage

**Demo App:**
```bash
streamlit run app.py
```

**Python API:**
```python
import cv2
from retina_seg.preprocessing.pipeline import run as preprocess
from retina_seg.enhancement.pipeline import run as enhance
from retina_seg.segmentation.pipeline import run as segment
from retina_seg.evaluation.metrics import evaluate

img_rgb = cv2.cvtColor(cv2.imread("fundus.jpg"), cv2.COLOR_BGR2RGB)
corrected, fov_mask = preprocess(img_rgb)
enhanced = enhance(corrected)
segmented = segment(enhanced, fov_mask)
```

## Run All Tests

```bash
pytest tests/ -v
```

## Target Metrics (DRIVE test set)

| SE | SP | ACC | AUC |
|----|----|-----|-----|
| ~0.72 | ~0.98 | ~0.957 | ~0.975 |
```

- [ ] **Step 5: Commit**

```bash
git add data/ README.md
git commit -m "docs: dataset download instructions and project README"
```

---

## Task 11: Full Test Suite Pass

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests pass. If any fail, fix before proceeding.

- [ ] **Step 2: Verify app starts cleanly**

```bash
python -c "
from retina_seg.preprocessing.pipeline import run as preprocess
from retina_seg.enhancement.pipeline import run as enhance
from retina_seg.segmentation.pipeline import run as segment
import numpy as np, cv2
img = np.zeros((256, 256, 3), dtype=np.uint8)
cv2.circle(img, (128, 128), 100, (180, 150, 120), -1)
img[60:65, 30:220] = (80, 60, 50)
c, m = preprocess(img)
e = enhance(c)
s = segment(e, m)
print('Pipeline OK, output unique values:', set(np.unique(s)))
"
```

Expected: `Pipeline OK, output unique values: {0, 255}`

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: verify full pipeline smoke test passes"
```
