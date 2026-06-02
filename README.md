# Retinal Vessel Segmentation — CLAHE + Matched Filter

Segmentasi pembuluh darah retina menggunakan pipeline image processing (CLAHE + Matched Filter) dan klasifikasi anomali retina menggunakan Machine Learning (SVM / Random Forest).

## Pipeline

```
RGB Image → Green Channel → FOV Mask → Illumination Correction
         → CLAHE → Gaussian → Matched Filter (12 orientations)
         → Otsu Thresholding → Morphological Post-processing
         → Binary Vessel Mask
```

**Klasifikasi Anomali (ML):**
```
Vessel Mask → Ekstrak Fitur Vessel → SVM / Random Forest → Normal / Abnormal (DR)

Fitur:
  • Vessel Density       — rasio pixel vessel terhadap FOV
  • Mean Vessel Width    — rata-rata lebar vessel via skeletonization
  • Tortuosity           — rasio panjang lengkung / jarak lurus per segmen
  • Branching Points     — kepadatan titik percabangan vessel
  • Fractal Dimension    — kompleksitas pola vessel (box-counting)
```

## Struktur Proyek

```
src/retina_seg/
├── dataset/        — loader untuk CHASE_DB1 dan APTOS 2019
├── preprocessing/  — green channel, FOV mask, illumination correction
├── enhancement/    — CLAHE, Gaussian, Matched Filter
├── segmentation/   — Otsu thresholding, morphological post-processing
├── evaluation/     — metrik SE/SP/ACC/AUC, overlay visualisasi
└── ml/             — ekstrak fitur vessel, train/predict SVM & RF
app.py              — Streamlit demo (3 tab)
tests/              — 28 unit tests
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Dataset Setup

**Segmentasi vessel (batch evaluation):**
- CHASE_DB1 — lihat `data/CHASE_DB1/README.md`

**Klasifikasi anomali (ML training):**
- APTOS 2019 Blindness Detection — download dari:
  `https://www.kaggle.com/competitions/aptos2019-blindness-detection/data`
- Ekstrak ke `data/APTOS2019/` dengan struktur:
  ```
  data/APTOS2019/
  ├── train_images/   ← ~3.662 gambar .png
  └── train.csv       ← kolom: id_code, diagnosis (0–4)
  ```

## Demo App

```bash
streamlit run app.py
```

| Tab | Fungsi |
|-----|--------|
| **Single Image** | Upload gambar → lihat tiap stage pipeline → download segmentasi |
| **Anomaly Classification (ML)** | Train SVM/RF pakai APTOS 2019 → prediksi Normal/Abnormal |
| **Batch Evaluation** | Evaluasi CLAHE+MF pada seluruh CHASE_DB1 → tabel SE/SP/ACC/AUC |

## Python API

```python
import cv2
from retina_seg.preprocessing.pipeline import run as preprocess
from retina_seg.enhancement.pipeline import run as enhance
from retina_seg.segmentation.pipeline import run as segment
from retina_seg.ml.pipeline import extract_image_features, train, predict

img_rgb = cv2.cvtColor(cv2.imread("fundus.jpg"), cv2.COLOR_BGR2RGB)

# Segmentasi vessel
corrected, fov_mask = preprocess(img_rgb)
enhanced = enhance(corrected)
vessel_mask = segment(enhanced, fov_mask)

# Klasifikasi anomali (setelah model ditraining)
features = extract_image_features(vessel_mask, fov_mask)  # shape (5,)
# model, scaler = train(X, y, method="random_forest")
# result = predict(model, scaler, vessel_mask, fov_mask)
# → {"label": 1, "label_str": "Abnormal (DR)", "probability": 0.87, ...}
```

## Run Tests

```bash
pytest tests/ -v
```

## Hasil Evaluasi — CHASE_DB1 (28 gambar)

| Metrik | Mean | Std |
|--------|------|-----|
| SE (Sensitivity) | 0.7687 | ±0.0542 |
| SP (Specificity) | 0.8463 | ±0.0490 |
| ACC | 0.8380 | ±0.0399 |
| AUC | 0.8560 | ±0.0191 |
