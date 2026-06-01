# Retinal Vessel Segmentation — Design Spec

**Date:** 2026-06-01
**Project:** Peningkatan Kontras dan Segmentasi Pembuluh Darah Retina Menggunakan CLAHE dan Matched Filter
**Course:** Pengolahan Citra Digital (PCD) — Final Project

---

## Overview

Build a pure image processing pipeline (no deep learning) to segment retinal blood vessels from fundus images. Pipeline runs on CPU without GPU. Evaluated on three standard benchmark datasets. A Streamlit web app provides the demo interface.

---

## Repo Structure

```
retina-vessel-seg/
├── src/
│   └── retina_seg/
│       ├── __init__.py
│       ├── dataset/
│       │   ├── __init__.py
│       │   ├── loader.py        # DRIVE, STARE, CHASE_DB1 loaders
│       │   └── utils.py         # path helpers, image I/O
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   └── pipeline.py      # green channel, FOV mask, illumination correction
│       ├── enhancement/
│       │   ├── __init__.py
│       │   └── pipeline.py      # CLAHE, Gaussian, Matched Filter
│       ├── segmentation/
│       │   ├── __init__.py
│       │   └── pipeline.py      # Otsu thresholding, morphological post-processing
│       ├── ml/
│       │   ├── __init__.py
│       │   └── pipeline.py      # feature extraction, SMOTE, SVM/RF (optional)
│       └── evaluation/
│           ├── __init__.py
│           └── metrics.py       # SE, SP, ACC, AUC, ROC curve, overlay viz
├── data/
│   ├── DRIVE/
│   ├── STARE/
│   └── CHASE_DB1/
├── docs/
│   └── superpowers/specs/
├── app.py                       # Streamlit demo
├── requirements.txt
└── README.md
```

---

## Module Interfaces

Every pipeline module exposes a single primary function that takes a numpy image array and returns a numpy image array, making them easy to chain and test independently.

| Module | Primary function | Input | Output |
|--------|-----------------|-------|--------|
| `dataset.loader` | `load_drive(root)` / `load_stare(root)` / `load_chase(root)` | path string | list of `{"image", "mask", "gt1", "gt2"}` dicts |
| `preprocessing.pipeline` | `run(img_rgb, fov_mask)` | RGB uint8, binary mask | corrected grayscale float |
| `enhancement.pipeline` | `run(img)` | grayscale float | vessel-enhanced float |
| `segmentation.pipeline` | `run(img, fov_mask)` | vessel-enhanced float, binary mask | binary segmentation uint8 |
| `evaluation.metrics` | `evaluate(pred, gt, fov_mask)` | binary arrays | dict with SE, SP, ACC, AUC |

---

## Pipeline Stages (Tahap 1–8, no training required)

### Pre-processing

**Stage 1 — Green Channel Extraction**
Extract the green channel from RGB: `I_green = img[:, :, 1]`. Green channel gives the highest vessel-to-background contrast in fundus images.

**Stage 2 — FOV Mask**
Generate a circular field-of-view mask automatically: convert to grayscale, threshold at a low value (e.g., > 20), apply morphological closing to fill holes, then erode slightly. For DRIVE, a pre-supplied mask is used directly.

**Stage 3 — Illumination Correction**
Estimate and subtract non-uniform background:
```
background = morphological_open(I_green, disk(radius=70))
I_corrected = I_green - background
```
Clip and normalize to [0, 255]. Apply FOV mask (set outside-FOV pixels to mean value).

### Enhancement

**Stage 4 — CLAHE**
Apply Contrast Limited Adaptive Histogram Equalization:
- `clipLimit = 2.0`
- `tileGridSize = (8, 8)`

**Stage 5 — Gaussian Smoothing**
Apply Gaussian filter with σ=1 to reduce high-frequency noise before vessel enhancement.

**Stage 6 — Matched Filter**
Model vessel cross-section as a Gaussian profile:
```
h(x, y) = -exp(-x² / 2σ²)   for |y| ≤ L/2
```
Parameters: σ=2, L=9. Apply at 12 orientations (0°, 15°, 30°, ..., 165°). Take the maximum response across all orientations at each pixel.

### Segmentation

**Stage 7 — Otsu Thresholding**
Apply Otsu's method to the vessel-enhanced image inside the FOV mask only. This automatically finds the optimal threshold separating vessel vs non-vessel pixels.

**Stage 8 — Morphological Post-processing**
1. Opening (disk r=1) — remove isolated noise pixels
2. Closing (disk r=1) — close small gaps in vessels
3. Remove small objects — delete connected components with area < 50 pixels

---

## ML Pipeline (Tahap 9–12, optional — requires ground truth)

Feature vector per pixel: [intensity, matched_filter_response, Frangi_response, gradient_magnitude, local_texture_LBP]. Ground truth labels from manual annotations. SMOTE to balance vessel (~10-15% of pixels) vs background. Train SVM (RBF kernel) and Random Forest. Save models with joblib. This module is decoupled from the main pipeline and run separately.

---

## Evaluation

Pixel-level metrics computed inside FOV mask only:

| Metric | Formula |
|--------|---------|
| Sensitivity (SE) | TP / (TP + FN) |
| Specificity (SP) | TN / (TN + FP) |
| Accuracy (ACC) | (TP + TN) / Total |
| AUC | Area under ROC curve |

Visualization output: side-by-side comparison of input, predicted segmentation, and ground truth with colored overlay (TP=green, FP=red, FN=blue).

---

## Streamlit Demo App (`app.py`)

Two tabs:

**Tab 1 — Single Image Pipeline**
- Upload any fundus image (JPG/PNG) or select from loaded dataset
- Select dataset type for correct FOV mask handling
- Run pipeline button → display intermediate results at each stage:
  - Original RGB, Green channel, FOV mask, Illumination-corrected
  - CLAHE output, Gaussian output, Matched filter response
  - Binary segmentation output
- Download result as PNG

**Tab 2 — Batch Evaluation**
- Select dataset (DRIVE / STARE / CHASE_DB1) and point to data folder
- Run pipeline on all images → show aggregate metrics table (SE, SP, ACC, AUC)
- Show ROC curve plot

---

## Datasets

| Dataset | Images | Resolution | Notes |
|---------|--------|-----------|-------|
| DRIVE | 40 (20 train, 20 test) | 564×584 px | FOV mask provided; 2 manual annotations |
| STARE | 20 | 605×700 px | Some with pathology |
| CHASE_DB1 | 28 | 999×960 px | School children; thinner vessels |

Datasets are not included in the repo. `data/` contains README with download instructions for each.

---

## Dependencies

```
opencv-python
scikit-image
scikit-learn
imbalanced-learn   # SMOTE
numpy
matplotlib
streamlit
joblib
Pillow
```

---

## Target Metrics (from reference papers)

| Dataset | SE | SP | ACC |
|---------|----|----|-----|
| DRIVE | ~0.72 | ~0.98 | ~0.957 |
| STARE | ~0.72 | ~0.97 | ~0.956 |
