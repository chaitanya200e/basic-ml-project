"""Streamlit app: predict a laptop's price.

    streamlit run app.py
"""
import json

import pandas as pd
import streamlit as st

from common import (
    CATEGORICAL, ENCODER_FILE, FEATURES_FILE, METRICS_FILE, MODEL_FILE, SCALER_FILE,
    compute_ppi, load_artifacts, predict_price,
)

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

RESOLUTIONS = {
    "1366 x 768 (HD)": (1366, 768),
    "1600 x 900 (HD+)": (1600, 900),
    "1920 x 1080 (Full HD)": (1920, 1080),
    "1920 x 1200": (1920, 1200),
    "2560 x 1440 (QHD)": (2560, 1440),
    "2560 x 1600": (2560, 1600),
    "2880 x 1800": (2880, 1800),
    "3200 x 1800 (QHD+)": (3200, 1800),
    "3840 x 2160 (4K)": (3840, 2160),
}
HDD_SIZES = [0, 500, 1000, 2000]
SSD_SIZES = [0, 128, 256, 512, 1000]


@st.cache_resource
def load_model_files():
    return load_artifacts()          # model + scaler + label encoder


def load_options(artifacts):
    # Dropdown values come straight from the label encoder, so the app can only offer
    # categories the model actually knows.
    known = {col: [str(v) for v in cats] for col, cats in zip(CATEGORICAL, artifacts["encoder"].categories_)}
    df = pd.read_csv(FEATURES_FILE)
    return {
        "company": known["Company"],
        "type": known["TypeName"],
        "cpu": known["CPU_name"],
        "gpu": known["Gpu_brand"],
        "os": known["OpSys"],
        "ram": sorted(int(r) for r in df["Ram"].unique()),
        "weight": round(float(df["Weight"].median()), 2),
    }


@st.cache_data
def load_metrics():
    return json.loads(METRICS_FILE.read_text()) if METRICS_FILE.exists() else {}


st.title("💻 Laptop Price Predictor")

if not all(f.exists() for f in (MODEL_FILE, SCALER_FILE, ENCODER_FILE)):
    st.error("Model files not found in data/artifacts/. Run `python main.py` first to train the model.")
    st.stop()

artifacts = load_model_files()
opts = load_options(artifacts)
metrics = load_metrics()

col1, col2 = st.columns(2)
with col1:
    company = st.selectbox("Brand", opts["company"], index=opts["company"].index("Dell") if "Dell" in opts["company"] else 0)
    laptop_type = st.selectbox("Type", opts["type"])
    ram = st.selectbox("RAM (GB)", opts["ram"], index=opts["ram"].index(8) if 8 in opts["ram"] else 0)
    weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=opts["weight"], step=0.05)
    os_name = st.selectbox("Operating system", opts["os"])
with col2:
    cpu = st.selectbox("Processor", opts["cpu"])
    gpu = st.selectbox("GPU brand", opts["gpu"])
    ssd = st.selectbox("SSD (GB)", SSD_SIZES, index=2)
    hdd = st.selectbox("HDD (GB)", HDD_SIZES, index=0)

st.subheader("Screen")
col3, col4 = st.columns(2)
with col3:
    inches = st.number_input("Screen size (inches)", min_value=10.0, max_value=18.4, value=15.6, step=0.1)
    resolution = st.selectbox("Resolution", list(RESOLUTIONS), index=2)
with col4:
    touchscreen = st.checkbox("Touchscreen")
    ips = st.checkbox("IPS panel")

if st.button("Predict price", type="primary"):
    x_res, y_res = RESOLUTIONS[resolution]
    price = predict_price(
        artifacts,
        Company=company,
        TypeName=laptop_type,
        OpSys=os_name,
        CPU_name=cpu,
        Gpu_brand=gpu,
        Ram=ram,
        Weight=weight,
        TouchScreen=int(touchscreen),
        IPS=int(ips),
        PPI=compute_ppi(x_res, y_res, inches),
        HDD=hdd,
        SSD=ssd,
    )
    st.success(f"Estimated price: ₹ {price:,.0f}")
    if metrics:
        st.caption(
            f"Model: {metrics['model']} · typical error on unseen laptops ≈ "
            f"₹ {metrics['test_mae_price']:,.0f} · R² = {metrics['test_r2_price']:.2f}"
        )
