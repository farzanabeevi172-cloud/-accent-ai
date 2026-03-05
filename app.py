import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from streamlit_mic_recorder import mic_recorder
import librosa
import tempfile
from datetime import datetime

# ---------------------------
# CONFIG
# ---------------------------
API_URL = "https://accent-backend.onrender.com/predict"
st.set_page_config(page_title="AccentAI Pro", page_icon="🎙", layout="wide")

# ---------------------------
# SESSION STATE
# ---------------------------
if "page" not in st.session_state:
    st.session_state.page = "welcome"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "users" not in st.session_state:
    st.session_state.users = {"test": "test"}
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------
# STYLING
# ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #050a12;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,180,255,0.07) 0%, transparent 70%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(0,80,180,0.06) 0%, transparent 60%);
    color: #e2eaf5;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }

/* ── Typography ── */
h1, h2, h3, .syne { font-family: 'Syne', sans-serif !important; }

/* ── Welcome Page ── */
.hero-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 80vh;
    text-align: center;
    gap: 1.4rem;
}
.hero-badge {
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #00c4f0;
    border: 1px solid rgba(0,196,240,0.25);
    border-radius: 100px;
    padding: 5px 16px;
    margin-bottom: 0.5rem;
    background: rgba(0,196,240,0.05);
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(3rem, 7vw, 5.5rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.03em;
    color: #ffffff;
    margin: 0;
}
.hero-title span {
    background: linear-gradient(135deg, #00c4f0 0%, #0072ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1.15rem;
    font-weight: 300;
    color: #7a92b0;
    max-width: 480px;
    line-height: 1.7;
    margin: 0;
}
.hero-orb {
    position: relative;
    width: 160px;
    height: 160px;
    margin: 1.5rem auto 0.5rem;
}
.hero-orb-inner {
    width: 160px; height: 160px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, rgba(0,196,240,0.25), rgba(0,114,255,0.08) 60%, transparent);
    border: 1px solid rgba(0,196,240,0.2);
    display: flex; align-items: center; justify-content: center;
    font-size: 56px;
    animation: pulse-orb 3s ease-in-out infinite;
    box-shadow: 0 0 60px rgba(0,180,255,0.12), inset 0 0 40px rgba(0,180,255,0.06);
}
@keyframes pulse-orb {
    0%, 100% { transform: scale(1); box-shadow: 0 0 60px rgba(0,180,255,0.12); }
    50% { transform: scale(1.04); box-shadow: 0 0 90px rgba(0,180,255,0.2); }
}
.hero-rings {
    position: absolute; inset: -20px;
    border-radius: 50%;
    border: 1px solid rgba(0,196,240,0.08);
    animation: ring-expand 3s ease-in-out infinite;
}
.hero-rings:nth-child(2) { inset: -40px; animation-delay: 0.5s; border-color: rgba(0,196,240,0.05); }
@keyframes ring-expand {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.05); }
}

/* ── Section Headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,196,240,0.2), transparent);
    margin-left: 8px;
}

/* ── Cards ── */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.8rem;
    backdrop-filter: blur(12px);
    transition: border-color 0.3s;
}
.glass-card:hover {
    border-color: rgba(0,196,240,0.18);
}

/* ── Login/Register Page ── */
.auth-wrap {
    max-width: 900px;
    margin: 2rem auto;
}
.auth-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #fff;
    text-align: center;
    margin-bottom: 0.4rem;
}
.auth-subtitle {
    text-align: center;
    color: #7a92b0;
    font-size: 0.95rem;
    margin-bottom: 2.5rem;
}
.auth-divider {
    width: 1px;
    background: linear-gradient(180deg, transparent, rgba(0,196,240,0.2), transparent);
    margin: 0 1.5rem;
}

/* ── Input Styling ── */
.stTextInput > label {
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #4a6a8a !important;
    margin-bottom: 4px !important;
}
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2eaf5 !important;
    padding: 10px 14px !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(0,196,240,0.5) !important;
    box-shadow: 0 0 0 3px rgba(0,196,240,0.08) !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    border: 1px solid rgba(0,196,240,0.4) !important;
    background: linear-gradient(135deg, rgba(0,196,240,0.12), rgba(0,114,255,0.08)) !important;
    color: #00c4f0 !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,196,240,0.22), rgba(0,114,255,0.16)) !important;
    border-color: rgba(0,196,240,0.7) !important;
    color: #fff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(0,180,255,0.15) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Dashboard Header ── */
.dash-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.2rem 1.6rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    margin-bottom: 2rem;
}
.dash-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    color: #fff;
}
.dash-logo span {
    background: linear-gradient(135deg, #00c4f0, #0072ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.dash-user {
    font-size: 0.8rem;
    color: #4a6a8a;
    letter-spacing: 0.08em;
    font-weight: 300;
}

/* ── Result Display ── */
.result-card {
    background: linear-gradient(135deg, rgba(0,196,240,0.06), rgba(0,114,255,0.04));
    border: 1px solid rgba(0,196,240,0.15);
    border-radius: 14px;
    padding: 1.4rem;
    text-align: center;
    margin-top: 1.2rem;
}
.result-accent {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.01em;
}
.result-flag { font-size: 2rem; margin-bottom: 4px; }
.result-conf {
    font-size: 0.78rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #4a6a8a;
    margin-top: 0.8rem;
}
.conf-track {
    width: 100%;
    height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    margin-top: 8px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    background: linear-gradient(90deg, #00c4f0, #0072ff);
    border-radius: 100px;
    transition: width 1s ease;
}

/* ── History ── */
.history-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px;
    margin-bottom: 8px;
    transition: border-color 0.2s;
}
.history-item:hover { border-color: rgba(0,196,240,0.15); }
.history-flag { font-size: 1.3rem; }
.history-accent {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    color: #fff;
    font-size: 0.95rem;
}
.history-meta { color: #4a6a8a; font-size: 0.78rem; margin-top: 2px; }
.history-conf {
    margin-left: auto;
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #00c4f0;
}
.history-empty {
    text-align: center;
    color: #2a3f55;
    font-size: 0.9rem;
    padding: 2rem;
    border: 1px dashed rgba(255,255,255,0.06);
    border-radius: 12px;
}

/* ── File uploader ── */
.stFileUploader {
    background: rgba(255,255,255,0.02) !important;
    border: 1px dashed rgba(0,196,240,0.2) !important;
    border-radius: 12px !important;
}
.stFileUploader label { color: #7a92b0 !important; font-size: 0.85rem !important; }

/* ── Audio player ── */
audio { width: 100% !important; border-radius: 8px !important; margin-top: 6px; }

/* ── Alerts ── */
.stSuccess, .stError, .stWarning {
    border-radius: 10px !important;
    border: none !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
}

/* ── Subheaders ── */
.stMarkdown h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.05em !important;
    color: #fff !important;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 2.5rem 0 1rem;
    color: #1e3048;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    font-weight: 300;
}

/* ── Matplotlib figure bg ── */
.stPlot { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# HELPERS
# ---------------------------
def login_user(username, password):
    return username in st.session_state.users and st.session_state.users[username] == password

def register_user(username, password):
    if username in st.session_state.users:
        return False
    st.session_state.users[username] = password
    return True

def show_waveform(audio_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            temp_name = tmp.name
        y, sr = librosa.load(temp_name, sr=None)
        fig, ax = plt.subplots(figsize=(5, 0.9))
        fig.patch.set_facecolor('#0a1520')
        ax.set_facecolor('#0a1520')
        ax.plot(y, color="#00c4f0", linewidth=0.6, alpha=0.85)
        # Gradient-ish fill
        ax.fill_between(range(len(y)), y, alpha=0.12, color="#00c4f0")
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        st.pyplot(fig)
        plt.close(fig)
    except:
        pass

def get_flag(accent):
    return {"Indian": "🇮🇳", "UK": "🇬🇧", "US": "🇺🇸"}.get(accent, "🌐")

def show_result(result, key_prefix=""):
    if "message" in result:
        st.warning(result["message"])
        return
    accent = result["predicted_accent"]
    confidence = result["confidence_score"]
    flag = get_flag(accent)
    st.session_state.history.append({
        "accent": accent,
        "confidence": confidence,
        "time": datetime.now().strftime("%H:%M:%S")
    })
    st.markdown(f"""
    <div class="result-card">
        <div class="result-flag">{flag}</div>
        <div class="result-accent">{accent}</div>
        <div class="result-conf">Confidence Score</div>
        <div class="conf-track">
            <div class="conf-fill" style="width:{confidence}%"></div>
        </div>
        <div style="font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:700;
                    color:#00c4f0; margin-top:8px;">{confidence}%</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# PAGE: WELCOME
# ---------------------------
if st.session_state.page == "welcome":
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-badge">✦ Real-Time Accent Detection</div>
        <div class="hero-orb">
            <div class="hero-rings"></div>
            <div class="hero-rings"></div>
            <div class="hero-orb-inner">🎙</div>
        </div>
        <h1 class="hero-title">Accent<span>AI</span> Pro</h1>
        <p class="hero-sub">Identify speech accents in real time with high-confidence neural analysis. Upload audio or record live.</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([2, 1, 2])
    with col_c:
        if st.button("Get Started →"):
            st.session_state.page = "login"
            st.rerun()

    st.markdown('<div class="app-footer">© 2026 AccentAI Pro &nbsp;·&nbsp; Powered by deep audio intelligence</div>', unsafe_allow_html=True)

# ---------------------------
# PAGE: LOGIN / REGISTER
# ---------------------------
elif st.session_state.page == "login":
    st.markdown("""
    <div style="text-align:center; padding: 2rem 0 2.5rem;">
        <div class="hero-badge">✦ Authentication</div>
        <div class="auth-title" style="margin-top:1rem;">Welcome Back</div>
        <div class="auth-subtitle">Sign in to your account or create a new one below</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🔑 Sign In</div>', unsafe_allow_html=True)
        username = st.text_input("Username", key="login_user", placeholder="Enter your username")
        password = st.text_input("Password", type="password", key="login_pass", placeholder="••••••••")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Login", key="login_btn"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.page = "dashboard"
                st.rerun()
            else:
                st.error("⚠ Invalid credentials. Please try again.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">✨ Create Account</div>', unsafe_allow_html=True)
        new_user = st.text_input("New Username", key="reg_user", placeholder="Choose a username")
        new_pass = st.text_input("New Password", type="password", key="reg_pass", placeholder="Create a password")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Register", key="reg_btn"):
            if register_user(new_user, new_pass):
                st.success("✓ Account created! You can now sign in.")
            else:
                st.error("⚠ Username already exists.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="app-footer">© 2026 AccentAI Pro</div>', unsafe_allow_html=True)

# ---------------------------
# PAGE: DASHBOARD
# ---------------------------
elif st.session_state.page == "dashboard" and st.session_state.logged_in:

    # Header bar
    col_logo, col_logout = st.columns([5, 1])
    with col_logo:
        st.markdown("""
        <div class="dash-header">
            <div class="dash-logo">Accent<span>AI</span> Pro</div>
            <div class="dash-user">🟢 &nbsp;Active Session</div>
        </div>
        """, unsafe_allow_html=True)
    with col_logout:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Sign Out"):
            st.session_state.logged_in = False
            st.session_state.page = "welcome"
            st.session_state.history = []
            st.rerun()

    # ── Main Analysis Columns ──
    col1, col2 = st.columns(2, gap="large")

    # --- UPLOAD AUDIO ---
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📂 Upload Audio</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#4a6a8a; font-size:0.85rem; margin-bottom:1rem;">Supported format: .wav</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["wav"], key="upload", label_visibility="collapsed")
        if uploaded_file:
            audio_bytes = uploaded_file.read()
            st.audio(audio_bytes)
            show_waveform(audio_bytes)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            if st.button("Analyze Audio", key="upload_btn"):
                with st.spinner("Analyzing..."):
                    try:
                        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
                        response = requests.post(API_URL, files=files)
                        if response.status_code == 200:
                            show_result(response.json(), key_prefix="upload")
                    except:
                        st.error("⚠ Could not connect to prediction service.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- RECORD AUDIO ---
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🎙 Record Live</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#4a6a8a; font-size:0.85rem; margin-bottom:1rem;">Click Start to begin recording your voice</div>', unsafe_allow_html=True)
        audio = mic_recorder(start_prompt="▶ Start Recording", stop_prompt="⏹ Stop Recording", key="recorder")
        if audio:
            st.audio(audio["bytes"])
            show_waveform(audio["bytes"])
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            if st.button("Analyze Recording", key="record_btn"):
                with st.spinner("Analyzing..."):
                    try:
                        files = {"file": ("live.wav", audio["bytes"], "audio/wav")}
                        response = requests.post(API_URL, files=files)
                        if response.status_code == 200:
                            show_result(response.json(), key_prefix="record")
                    except:
                        st.error("⚠ Could not connect to prediction service.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── History Section ──
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📋 Prediction History</div>', unsafe_allow_html=True)

    if st.session_state.history:
        for item in reversed(st.session_state.history):
            flag = get_flag(item["accent"])
            st.markdown(f"""
            <div class="history-item">
                <div class="history-flag">{flag}</div>
                <div>
                    <div class="history-accent">{item['accent']}</div>
                    <div class="history-meta">Detected at {item['time']}</div>
                </div>
                <div class="history-conf">{item['confidence']}%</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="history-empty">
            No predictions yet. Upload or record audio to get started.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="app-footer">© 2026 AccentAI Pro &nbsp;·&nbsp; Powered by deep audio intelligence</div>', unsafe_allow_html=True)