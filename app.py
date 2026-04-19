import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import sounddevice as sd

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="CGRI AI Dashboard", layout="wide")

st.title("🌿 AI-Integrated Context-Aware Green Resilience Index (CGRI)")
st.write("Computer Vision + Real-Time Acoustic Sensing + Environmental Intelligence")

st.markdown("---")

# =========================================================
# FUNCTION: REAL-TIME NOISE MEASUREMENT
# =========================================================
def measure_noise(duration=3, fs=44100):
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    
    rms = np.sqrt(np.mean(recording**2))
    db = 20 * np.log10(rms + 1e-6)
    
    estimated_db = np.clip(db + 100, 30, 90)
    return round(float(estimated_db), 2)

# =========================================================
# SECTION 1 — AI GREEN COVER DETECTION
# =========================================================
st.header("🧠 Automatic Green Cover Detection")

uploaded_image = st.file_uploader("Upload Campus Area Image", type=["jpg", "jpeg", "png"])

green_percentage = None

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    green_pixels = np.count_nonzero(mask)
    total_pixels = mask.size
    green_percentage = (green_pixels / total_pixels) * 100

    st.subheader(f"🌿 Detected Green Cover: {green_percentage:.2f}%")
    st.image(mask, caption="Detected Vegetation Mask", use_container_width=True)

st.markdown("---")

# =========================================================
# SECTION 2 — CGRI CALCULATOR
# =========================================================
st.header("📊 CGRI Calculator")

col1, col2 = st.columns(2)

with col1:
    if green_percentage is not None:
        green_cover = st.slider("Green Cover (%)", 0, 100, int(green_percentage))
    else:
        green_cover = st.slider("Green Cover (%)", 0, 100, 60)

    shade = st.selectbox("Shade Level (1–3)", [1, 2, 3])
    duration = st.slider("Duration (minutes)", 10, 40, 20)

with col2:
    st.subheader("🔊 Noise Detection")

    if st.button("Measure Real-Time Noise (3 sec)"):
        measured_noise = measure_noise()
        st.session_state["noise_level"] = measured_noise
        st.success(f"Measured Noise Level: {measured_noise} dB")

    if "noise_level" not in st.session_state:
        st.session_state["noise_level"] = 60

    noise = st.slider("Noise Level (dB)", 30, 90, int(st.session_state["noise_level"]))

    crowd = st.selectbox("Crowd Density (1–3)", [1, 2, 3])
    exam = st.selectbox("Exam Period (0 = No, 1 = Yes)", [0, 1])

# Normalization
green_n = green_cover / 100
shade_n = (shade - 1) / 2
duration_n = (duration - 10) / 30
noise_n = (noise - 30) / 60
crowd_n = (crowd - 1) / 2
exam_n = exam

EC = np.mean([green_n, shade_n, duration_n])
SP = np.mean([noise_n, crowd_n, exam_n])
CGRI = EC - SP

st.subheader("📈 CGRI Result")

colA, colB, colC = st.columns(3)
colA.metric("Environmental Capacity (EC)", round(EC, 3))
colB.metric("Stress Pressure (SP)", round(SP, 3))
colC.metric("CGRI Score", round(CGRI, 3))

# Gauge
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=CGRI,
    title={'text': "CGRI Score"},
    gauge={
        'axis': {'range': [-1, 1]},
        'steps': [
            {'range': [-1, -0.1], 'color': "red"},
            {'range': [-0.1, 0.2], 'color': "yellow"},
            {'range': [0.2, 1], 'color': "lightgreen"}
        ],
    }
))
st.plotly_chart(fig, use_container_width=True)

# =========================================================
# SECTION 3 — OPTIMIZATION ENGINE
# =========================================================
st.markdown("---")
st.header("🚀 Optimization Recommendation Engine")

if CGRI < 0.2:
    st.warning("Resilience is not High. Suggested Improvements:")

    recommendations = []

    if green_cover < 70:
        recommendations.append("Increase green cover above 70%")
    if shade < 3:
        recommendations.append("Increase shade level to 3")
    if noise > 50:
        recommendations.append("Reduce noise level below 50 dB")
    if duration < 30:
        recommendations.append("Encourage stay duration ≥ 30 minutes")
    if crowd > 1:
        recommendations.append("Reduce crowd density")

    for rec in recommendations:
        st.write("•", rec)
else:
    st.success("This environment already exhibits High Resilience.")

# =========================================================
# SECTION 4 — DATASET UPLOAD & ADVANCED ANALYSIS
# =========================================================
st.markdown("---")
st.header("📁 Batch CGRI Analysis")

uploaded_csv = st.file_uploader("Upload CSV Dataset", type=["csv"], key="dataset")

if uploaded_csv is not None:

    df = pd.read_csv(uploaded_csv)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    required_cols = [
        "green_cover_pct",
        "noise_db",
        "shade_level",
        "crowd_density",
        "duration_minutes",
        "exam_period"
    ]

    if all(col in df.columns for col in required_cols):

        df["green_n"] = df["green_cover_pct"] / 100
        df["shade_n"] = (df["shade_level"] - 1) / 2
        df["duration_n"] = (df["duration_minutes"] - 10) / 30
        df["noise_n"] = (df["noise_db"] - 30) / 60
        df["crowd_n"] = (df["crowd_density"] - 1) / 2
        df["exam_n"] = df["exam_period"]

        df["EC"] = df[["green_n", "shade_n", "duration_n"]].mean(axis=1)
        df["SP"] = df[["noise_n", "crowd_n", "exam_n"]].mean(axis=1)
        df["CGRI"] = df["EC"] - df["SP"]

        df["Resilience_Level"] = pd.cut(
            df["CGRI"],
            bins=[-1, -0.1, 0.2, 1],
            labels=["Low", "Moderate", "High"]
        )

        # Histogram
        st.subheader("📊 CGRI Distribution")

        fig_hist = px.histogram(df, x="CGRI", nbins=30, opacity=0.8)

        mean_cgri = df["CGRI"].mean()
        std_cgri = df["CGRI"].std()
        skew_cgri = df["CGRI"].skew()

        fig_hist.add_vline(x=mean_cgri, line_dash="dash")
        fig_hist.add_vline(x=-0.1, line_dash="dot", line_color="red")
        fig_hist.add_vline(x=0.2, line_dash="dot", line_color="green")

        st.plotly_chart(fig_hist, use_container_width=True)

        # Statistics
        st.subheader("📈 Statistical Summary")

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean", f"{mean_cgri:.3f}")
        col2.metric("Std Dev", f"{std_cgri:.3f}")
        col3.metric("Skewness", f"{skew_cgri:.3f}")

        low_pct = (df["Resilience_Level"] == "Low").mean() * 100
        mod_pct = (df["Resilience_Level"] == "Moderate").mean() * 100
        high_pct = (df["Resilience_Level"] == "High").mean() * 100

        st.write(f"Low: {low_pct:.2f}%")
        st.write(f"Moderate: {mod_pct:.2f}%")
        st.write(f"High: {high_pct:.2f}%")

    else:
        st.error("Dataset must contain required columns.")