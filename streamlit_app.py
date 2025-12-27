import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import requests
import time

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="✍️ Air-Draw Digit Recognition",
    layout="wide"
)

# =========================================================
# CSS (UNCHANGED)
# =========================================================
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #020617);
    border-right: 1px solid rgba(34,197,94,0.35);
}
section[data-testid="stSidebar"] h1 {
    color: #38BDF8;
    text-align: center;
    font-weight: 800;
}
section[data-testid="stSidebar"] label {
    color: #E5E7EB !important;
}
section[data-testid="stSidebar"] .stRadio label {
    background: #0F172A;
    padding: 10px 14px;
    border-radius: 10px;
    margin-bottom: 8px;
    border: 1px solid rgba(34,197,94,0.25);
}
section[data-testid="stSidebar"] .stRadio label:hover {
    border-color: #22C55E;
    box-shadow: 0 0 16px rgba(34,197,94,0.45);
}

.page-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #38BDF8;
    margin-bottom: 25px;
}

.tile {
    background: radial-gradient(circle at top left, #020617, #0B1220 55%, #020617);
    padding: 28px;
    border-radius: 20px;
    border: 1px solid rgba(34,197,94,0.35);
    box-shadow:
        0 0 18px rgba(34,197,94,0.22),
        0 0 45px rgba(34,197,94,0.12),
        inset 0 0 12px rgba(34,197,94,0.05);
    margin-bottom: 26px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.tile:hover {
    transform: translateY(-5px);
    box-shadow:
        0 0 26px rgba(34,197,94,0.45),
        0 0 70px rgba(34,197,94,0.25),
        inset 0 0 18px rgba(34,197,94,0.08);
}
.tile h3 {
    margin-top: 0;
    color: #38BDF8;
    font-weight: 800;
}
.tile p {
    color: #E5E7EB;
    font-size: 15.5px;
    line-height: 1.7;
}

.result-tile {
    background: linear-gradient(145deg, #020617, #0F172A);
    padding: 30px;
    border-radius: 20px;
    border: 1px solid rgba(34,197,94,0.35);
    text-align: center;
    box-shadow:
        0 0 25px rgba(34,197,94,0.35),
        0 0 60px rgba(34,197,94,0.25);
    margin-top: 30px;
}
.result-title {
    font-size: 20px;
    font-weight: 700;
    color: #38BDF8;
    margin-bottom: 10px;
}
.result-value {
    font-size: 52px;
    font-weight: 900;
    color: #E5E7EB;
}
.result-confidence {
    font-size: 16px;
    color: #CBD5F5;
}

.live-btn button {
    height: 3.2em;
    font-size: 18px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# CONFIGURATION
# =========================================================
BASE_URL = "http://192.0.0.4:8080"
FS = 100
DURATION = 2
T = 200
FEATURES = ["ax", "ay", "az", "gx", "gy", "gz"]

# =========================================================
# LOAD MODEL & NORMALIZATION
# =========================================================
model = tf.keras.models.load_model("model/airdraw_model.keras")
norm = np.load("model/norm_stats.npz")
mean, std = norm["mean"], norm["std"]

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def resample_from_df(df, T=200):
    old_len = len(df)
    old_idx = np.linspace(0, old_len - 1, old_len)
    new_idx = np.linspace(0, old_len - 1, T)
    X = np.zeros((T, len(FEATURES)))
    for i, col in enumerate(FEATURES):
        X[:, i] = np.interp(new_idx, old_idx, df[col].values)
    return X

def safe_get(url):
    try:
        return requests.get(url)
    except requests.exceptions.ConnectionError:
        st.error("❌📡 Cannot connect to phone. Please check:\n"
                 "- Phone & PC are on same Wi-Fi\n"
                 "- Phyphox is running\n"
                 "- Remote access is enabled\n"
                 "- IP address is correct")
        return None
    except requests.exceptions.Timeout:
        st.error("⏳ Connection timed out. Phone not responding.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Network error: {e}")
        return None

def start_recording(): safe_get(f"{BASE_URL}/control?cmd=start")
def stop_recording(): safe_get(f"{BASE_URL}/control?cmd=stop")
def clear_recording(): safe_get(f"{BASE_URL}/control?cmd=clear")

def fetch_live_buffers():
    resp = safe_get(
        f"{BASE_URL}/get?"
        "accX=full&accY=full&accZ=full&"
        "gyrX=full&gyrY=full&gyrZ=full"
    )
    if resp is None:
        return 0, [], [], [], [], [], []

    data = resp.json()["buffer"]
    accX = data["accX"]["buffer"]
    accY = data["accY"]["buffer"]
    accZ = data["accZ"]["buffer"]
    gyrX = data["gyrX"]["buffer"]
    gyrY = data["gyrY"]["buffer"]
    gyrZ = data["gyrZ"]["buffer"]

    n = min(len(accX), len(accY), len(accZ),
            len(gyrX), len(gyrY), len(gyrZ))

    return n, accX[:n], accY[:n], accZ[:n], gyrX[:n], gyrY[:n], gyrZ[:n]

def record_digit():
    st.info("✋✍️ Get ready to draw the digit in the air.")
    st.caption("Tip: Hold the phone firmly and draw smoothly for better accuracy.")
    time.sleep(1)

    start_recording()
    st.warning("⏺️ Recording in progress. Please draw now.")
    time.sleep(DURATION)
    stop_recording()

    st.success("✅ Recording completed. Generating prediction...")

    n, ax, ay, az, gx, gy, gz = fetch_live_buffers()

    df = pd.DataFrame({
        "timestamp": np.arange(n) / FS,
        "ax": ax, "ay": ay, "az": az,
        "gx": gx, "gy": gy, "gz": gz
    })

    clear_recording()
    return df

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
page = st.sidebar.radio(
    "🧭 Navigation Pages👇",
    [
        "🏠 Home",
        "📁 Air-Draw Digit Prediction (CSV)",
        "🟢 Live Air-Draw (2s Start)",
        "ℹ️ About"
    ]
)

# =========================================================
# HOME PAGE
# =========================================================
if page == "🏠 Home":
    st.markdown("<div class='page-title'>✍️ Air-Draw Digit Recognition</div>", unsafe_allow_html=True)

    st.markdown(
    "<div style='text-align:center; margin-top:-10px; margin-bottom:20px; opacity:0.8; font-size:0.9rem;'>✨ Draw numbers in the air using your smartphone and let AI recognize them!</div>",
    unsafe_allow_html=True
)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="tile">
            <h3>📁 CSV Mode 📄</h3>
            <p>
            Draw a digit in the air using <b>Phyphox</b>, export the recorded
            sensor data as a CSV file, upload it here, and receive an
            instant digit prediction 🎯.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="tile">
            <h3>🟢 Live Mode ✋</h3>
            <p>
            Click start and draw a digit in the air for 2 seconds ⏱️.
            The system captures motion data in real time and predicts
            the digit automatically 🤖.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tile">
        <h3>🎯 Project Goal</h3>
        <p>
        To recognize digits written in the air using smartphone motion
        sensors and a deep learning model 🚀.
        </p>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# CSV MODE
# =========================================================
elif page == "📁 Air-Draw Digit Prediction (CSV)":
    st.markdown("<div class='page-title'>📁 CSV Digit Prediction</div>", unsafe_allow_html=True)
    st.caption("📌 Ensure the CSV file contains accelerometer and gyroscope data.")
    uploaded_file = st.file_uploader("📤 Upload an Air-Draw CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        X = resample_from_df(df, T)
        X = (X - mean) / std
        X = X[np.newaxis, :, :]

        preds = model.predict(X)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])

        st.markdown(f"""
        <div class="result-tile">
            <div class="result-title">🔢 Predicted Digit</div>
            <div class="result-value">✍️ {idx}</div>
            <div class="result-confidence">📊 Confidence level: {confidence:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(confidence)

# =========================================================
# LIVE MODE
# =========================================================
elif page == "🟢 Live Air-Draw (2s Start)":
    st.markdown("<div class='page-title'>🟢 Live Air-Draw Prediction</div>", unsafe_allow_html=True)
    st.caption("📱 Make sure your phone stays connected throughout recording.")

    st.markdown("<div class='live-btn'>", unsafe_allow_html=True)
    start = st.button("▶️ Start 2-Second Recording", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if start:
        buffer = record_digit()
        X = resample_from_df(buffer, T)
        X = (X - mean) / std
        X = X[np.newaxis, :, :]

        preds = model.predict(X)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])

        st.markdown(f"""
        <div class="result-tile">
            <div class="result-title">🔢 Predicted Digit</div>
            <div class="result-value">✍️ {idx}</div>
            <div class="result-confidence">📊 Confidence level: {confidence:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(confidence)

# =========================================================
# ABOUT PAGE
# =========================================================
elif page == "ℹ️ About":
    st.markdown("<div class='page-title'>ℹ️ About This Project</div>", unsafe_allow_html=True)
    st.caption("📘 Academic + real-world application of sensor-based deep learning")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="tile">
            <h3>📌 Overview</h3>
            <p>
            This system enables users to write digits in the air using a
            smartphone ✋📱. Motion sensor data is captured and classified
            using a deep learning model 🤖.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="tile">
            <h3>⚙️ Technical Stack</h3>
            <p>
            Sensors: Accelerometer & Gyroscope 🧭<br>
            Model: CNN + LSTM 🧠<br>
            Input Shape: (200, 6)<br>
            Output Classes: Digits 0–9 🔢
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tile">
        <h3>📱 Data Source</h3>
        <p>
        Data is collected using the Phyphox mobile application 📡
        with full sensor buffers and processed using Streamlit 🚀.
        </p>
    </div>
    """, unsafe_allow_html=True)