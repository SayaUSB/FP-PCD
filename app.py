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
from retina_seg.dataset.loader import load_chase, load_aptos
from retina_seg.ml.pipeline import (
    extract_image_features, train as ml_train, predict as ml_predict,
    save_model, load_model, FEATURE_NAMES,
)

st.set_page_config(page_title="Retinal Vessel Segmentation", layout="wide")
st.title("Segmentasi Pembuluh Darah Retina — CLAHE + Matched Filter")

tab1, tab2, tab3 = st.tabs(["Single Image", "Anomaly Classification (ML)", "Batch Evaluation"])

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
                segmented, _ = segment(enhanced, fov_mask, return_intermediates=True)

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
            st.download_button("Download Segmentation", buf.getvalue(), "segmentation.png", "image/png")

# ── Tab 2: Anomaly Classification ────────────────────────────────────────────
with tab2:
    st.header("Klasifikasi Anomali Retina (Normal vs Abnormal)")
    st.markdown(
        "Pipeline: segmentasi vessel → ekstrak fitur vessel → SVM/RF klasifikasikan "
        "gambar sebagai **Normal** atau **Abnormal (Diabetic Retinopathy)**."
    )

    # ── Training ─────────────────────────────────────────────────────────────
    st.subheader("1. Training Model")
    col_l, col_r = st.columns(2)
    with col_l:
        aptos_path = st.text_input(
            "Path folder APTOS 2019",
            placeholder="/home/user/data/APTOS2019",
            key="aptos_path",
        )
        ml_method = st.selectbox("Metode", ["random_forest", "svm"], key="ml_method")
    with col_r:
        max_train = st.slider(
            "Max gambar untuk training",
            min_value=50, max_value=2000, value=500, step=50,
            help="Lebih banyak = lebih akurat tapi lebih lambat",
        )
        st.markdown("**Fitur yang diekstrak dari vessel mask:**")
        for f in FEATURE_NAMES:
            st.caption(f"• {f}")

    if st.button("Train Model") and aptos_path:
        try:
            samples = load_aptos(aptos_path)
        except Exception as exc:
            st.error(f"Gagal load dataset: {exc}")
            samples = []

        if samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(samples), min(max_train, len(samples)), replace=False)
            selected = [samples[i] for i in idx]

            n_normal = sum(1 for s in selected if s["label"] == 0)
            n_abnormal = sum(1 for s in selected if s["label"] == 1)
            st.info(f"Training pada {len(selected)} gambar — Normal: {n_normal}, Abnormal: {n_abnormal}")

            all_features, all_labels = [], []
            progress = st.progress(0, text="Segmentasi + ekstrak fitur...")

            for i, s in enumerate(selected):
                corrected, fov_mask = preprocess(s["image"], s.get("mask"))
                enhanced = enhance(corrected)
                vessel_mask = segment(enhanced, fov_mask)
                feats = extract_image_features(vessel_mask, fov_mask)
                all_features.append(feats)
                all_labels.append(s["label"])
                progress.progress((i + 1) / len(selected), text=f"{i+1}/{len(selected)} gambar")

            X = np.vstack(all_features)
            y = np.array(all_labels)

            with st.spinner(f"Training {ml_method.upper()}..."):
                model, scaler = ml_train(X, y, method=ml_method)

            st.session_state["ml_model"] = model
            st.session_state["ml_scaler"] = scaler
            st.session_state["ml_method"] = ml_method
            st.success("Model berhasil dilatih!")

            # Feature importance (RF only)
            if ml_method == "random_forest":
                importances = model.feature_importances_
                fi_df = pd.DataFrame({
                    "Fitur": FEATURE_NAMES,
                    "Importance": importances,
                }).sort_values("Importance", ascending=False)
                st.subheader("Feature Importance")
                st.dataframe(fi_df.set_index("Fitur").round(4))

            # Save model button
            buf = io.BytesIO()
            import joblib
            joblib.dump({"model": model, "scaler": scaler}, buf)
            st.download_button(
                "Download Model (.pkl)",
                buf.getvalue(),
                "retina_model.pkl",
                "application/octet-stream",
            )

    # ── Prediction ───────────────────────────────────────────────────────────
    st.subheader("2. Prediksi Gambar Baru")

    # Load saved model
    uploaded_model = st.file_uploader("Atau load model yang sudah disimpan (.pkl)", type=["pkl"], key="model_upload")
    if uploaded_model:
        import joblib
        data = joblib.load(uploaded_model)
        st.session_state["ml_model"] = data["model"]
        st.session_state["ml_scaler"] = data["scaler"]
        st.session_state["ml_method"] = type(data["model"]).__name__
        st.success("Model berhasil dimuat.")

    if "ml_model" not in st.session_state:
        st.info("Train model terlebih dahulu atau load model yang sudah disimpan.")
    else:
        st.caption(f"Model aktif: **{st.session_state['ml_method']}**")
        uploaded_test = st.file_uploader(
            "Upload citra fundus",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            key="pred_upload",
        )

        if uploaded_test and st.button("Klasifikasi"):
            img_pil = Image.open(uploaded_test).convert("RGB")
            img_rgb = np.array(img_pil)

            with st.spinner("Memproses..."):
                corrected, fov_mask = preprocess(img_rgb)
                enhanced = enhance(corrected)
                vessel_mask = segment(enhanced, fov_mask)
                result = ml_predict(
                    st.session_state["ml_model"],
                    st.session_state["ml_scaler"],
                    vessel_mask,
                    fov_mask,
                )

            cols = st.columns(2)
            cols[0].image(img_rgb, caption="Input", use_container_width=True)
            cols[1].image(vessel_mask, caption="Vessel Segmentation", use_container_width=True)

            label_color = "red" if result["label"] == 1 else "green"
            st.markdown(
                f"### Hasil: :{label_color}[{result['label_str']}]  "
                f"(P(abnormal) = {result['probability']:.3f})"
            )

            feat_df = pd.DataFrame(
                result["features"].items(), columns=["Fitur", "Nilai"]
            ).set_index("Fitur")
            st.dataframe(feat_df.round(4))

# ── Tab 3: Batch Evaluation ──────────────────────────────────────────────────
with tab3:
    st.header("Batch Evaluation — CHASE_DB1")
    st.markdown("Evaluasi pipeline CLAHE+MF pada seluruh gambar CHASE_DB1 dan tampilkan metrik agregat untuk laporan.")

    batch_path = st.text_input(
        "Path folder CHASE_DB1",
        placeholder="/home/user/data/CHASE_DB1",
        key="batch_path",
    )

    if st.button("Run Batch Evaluation") and batch_path:
        try:
            samples = load_chase(batch_path)
        except Exception as exc:
            st.error(f"Gagal load dataset: {exc}")
            samples = []

        if samples:
            rows = []
            progress = st.progress(0, text="Memproses...")
            for i, s in enumerate(samples):
                if s["gt1"] is None:
                    continue
                corrected, fov_mask = preprocess(s["image"], s.get("mask"))
                enhanced = enhance(corrected)
                segmented = segment(enhanced, fov_mask)
                metrics = evaluate(segmented, s["gt1"], fov_mask)
                auc_val, _, _ = compute_auc(enhanced, s["gt1"], fov_mask)
                rows.append({
                    "name": s["name"],
                    "SE": round(metrics["SE"], 4),
                    "SP": round(metrics["SP"], 4),
                    "ACC": round(metrics["ACC"], 4),
                    "AUC": round(auc_val, 4),
                })
                progress.progress((i + 1) / len(samples), text=f"{i+1}/{len(samples)} gambar")

            if rows:
                df = pd.DataFrame(rows).set_index("name")

                st.subheader("Hasil Per Gambar")
                st.dataframe(df, use_container_width=True)

                st.subheader("Statistik Agregat")
                agg = df.agg(["mean", "std", "min", "max"]).round(4)
                agg.index = ["Mean", "Std", "Min", "Max"]
                st.dataframe(agg, use_container_width=True)

                st.subheader("Contoh Overlay (gambar pertama)")
                s0 = samples[0]
                corrected0, fov0 = preprocess(s0["image"], s0.get("mask"))
                enhanced0 = enhance(corrected0)
                seg0 = segment(enhanced0, fov0)
                overlay0 = make_overlay(s0["image"], seg0, s0["gt1"], fov0)
                st.image(overlay0, caption=f"{s0['name']} — Hijau=TP, Merah=FP, Biru=FN", width=450)
