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
python3 -m venv .venv
source .venv/bin/activate
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
