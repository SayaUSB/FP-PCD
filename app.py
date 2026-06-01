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
        st.image(img_rgb, caption="Input Image", use_container_width=False, width=300)

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
