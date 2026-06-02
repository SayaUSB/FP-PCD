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
from retina_seg.ml.pipeline import extract_features, train as ml_train, predict as ml_predict

st.set_page_config(page_title="Retinal Vessel Segmentation", layout="wide")
st.title("Segmentasi Pembuluh Darah Retina — CLAHE + Matched Filter")

tab1, tab2 = st.tabs(["Single Image", "ML Segmentation"])

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

# ── Tab 2: ML Segmentation ───────────────────────────────────────────────────
with tab2:
    st.header("ML Segmentation (SVM / Random Forest)")
    st.markdown(
        "Train a pixel-level classifier on a retinal dataset, kemudian jalankan prediksi "
        "pada gambar baru. Fitur yang digunakan: green intensity, matched filter response, "
        "Frangi filter, gradient magnitude, LBP."
    )

    # ── Training section ─────────────────────────────────────────────────────
    st.subheader("1. Training")
    col_left, col_right = st.columns(2)
    with col_left:
        train_dataset = st.selectbox("Dataset untuk training", ["DRIVE", "STARE", "CHASE_DB1"])
        train_path = st.text_input(
            "Path folder dataset training",
            placeholder="/home/user/data/DRIVE",
            key="train_path",
        )
        train_split = st.radio(
            "Split (DRIVE only)", ["training", "test"], horizontal=True, key="train_split"
        )
    with col_right:
        ml_method = st.selectbox("Metode klasifikasi", ["svm", "random_forest"])
        max_samples = st.slider(
            "Max pixel samples per gambar (ribu)",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            help="Sampling acak agar training tidak terlalu lama. Lebih banyak = lebih akurat tapi lebih lambat.",
        )

    if st.button("Train Model") and train_path:
        loader_map = {
            "DRIVE": lambda p: load_drive(p, split=train_split),
            "STARE": load_stare,
            "CHASE_DB1": load_chase,
        }
        try:
            samples = loader_map[train_dataset](train_path)
        except Exception as exc:
            st.error(f"Gagal load dataset: {exc}")
            samples = []

        if samples:
            all_features, all_labels = [], []
            rng = np.random.default_rng(42)
            progress = st.progress(0, text="Mengekstrak fitur...")

            for i, s in enumerate(samples):
                if s["gt1"] is None:
                    continue
                corrected, fov_mask = preprocess(s["image"], s.get("mask"))
                enhanced = enhance(corrected)
                feats = extract_features(s["image"], enhanced, fov_mask)
                gt_flat = (s["gt1"][fov_mask > 0] > 0).astype(np.uint8)

                n = len(feats)
                k = min(n, max_samples * 1000)
                idx = rng.choice(n, k, replace=False)
                all_features.append(feats[idx])
                all_labels.append(gt_flat[idx])
                progress.progress((i + 1) / len(samples), text=f"Fitur {i+1}/{len(samples)}")

            X = np.vstack(all_features)
            y = np.concatenate(all_labels)
            st.info(f"Total sampel: {len(X):,} ({y.sum():,} vessel, {(1-y).sum():,} background)")

            with st.spinner(f"Training {ml_method.upper()}..."):
                model = ml_train(X, y, method=ml_method)

            st.session_state["ml_model"] = model
            st.session_state["ml_method"] = ml_method
            st.success("Model berhasil dilatih! Lanjut ke bagian prediksi di bawah.")

    # ── Prediction section ───────────────────────────────────────────────────
    st.subheader("2. Prediksi pada Gambar Baru")

    if "ml_model" not in st.session_state:
        st.info("Train model terlebih dahulu di bagian atas.")
    else:
        st.caption(f"Model aktif: **{st.session_state['ml_method'].upper()}**")
        uploaded_test = st.file_uploader(
            "Upload citra fundus untuk prediksi",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            key="ml_test_upload",
        )
        uploaded_gt = st.file_uploader(
            "Upload ground truth mask (opsional, untuk evaluasi)",
            type=["jpg", "jpeg", "png", "tif", "tiff", "gif"],
            key="ml_gt_upload",
        )

        if uploaded_test and st.button("Run ML Prediction"):
            img_pil = Image.open(uploaded_test).convert("RGB")
            img_rgb = np.array(img_pil)

            with st.spinner("Prediksi..."):
                corrected, fov_mask = preprocess(img_rgb)
                enhanced = enhance(corrected)

                # Image processing result (for comparison)
                ip_result = segment(enhanced, fov_mask)

                # ML result
                ml_result = ml_predict(
                    st.session_state["ml_model"], img_rgb, enhanced, fov_mask
                )

            st.subheader("Perbandingan Hasil")
            cols = st.columns(3)
            cols[0].image(img_rgb, caption="Input", use_container_width=True)
            cols[1].image(ip_result, caption="CLAHE+MF (Image Processing)", use_container_width=True)
            cols[2].image(ml_result, caption=f"ML ({st.session_state['ml_method'].upper()})", use_container_width=True)

            if uploaded_gt:
                gt_pil = Image.open(uploaded_gt).convert("L")
                gt_arr = (np.array(gt_pil) > 127).astype(np.uint8) * 255

                ip_metrics = evaluate(ip_result, gt_arr, fov_mask)
                ml_metrics = evaluate(ml_result, gt_arr, fov_mask)

                ip_auc, _, _ = compute_auc(enhanced, gt_arr, fov_mask)
                ml_auc, _, _ = compute_auc(
                    cv2.normalize(ml_result.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX),
                    gt_arr,
                    fov_mask,
                )

                st.subheader("Evaluasi Metrik")
                metric_df = pd.DataFrame(
                    {
                        "CLAHE+MF": {
                            "SE (Sensitivity)": round(ip_metrics["SE"], 4),
                            "SP (Specificity)": round(ip_metrics["SP"], 4),
                            "ACC": round(ip_metrics["ACC"], 4),
                            "AUC": round(ip_auc, 4),
                        },
                        f"ML ({st.session_state['ml_method'].upper()})": {
                            "SE (Sensitivity)": round(ml_metrics["SE"], 4),
                            "SP (Specificity)": round(ml_metrics["SP"], 4),
                            "ACC": round(ml_metrics["ACC"], 4),
                            "AUC": round(ml_auc, 4),
                        },
                    }
                )
                st.dataframe(metric_df)

                st.subheader("Overlay (ML Result)")
                overlay = make_overlay(img_rgb, ml_result, gt_arr, fov_mask)
                st.image(overlay, caption="Hijau=TP, Merah=FP, Biru=FN", width=400)

            buf = io.BytesIO()
            Image.fromarray(ml_result).save(buf, format="PNG")
            st.download_button(
                "Download ML Segmentation",
                buf.getvalue(),
                "ml_segmentation.png",
                "image/png",
            )
