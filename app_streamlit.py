import streamlit as st
import torch
import librosa
import numpy as np
import joblib
import tempfile
import os

from transformers import Wav2Vec2Model, Wav2Vec2Processor
from audio_recorder_streamlit import audio_recorder

st.set_page_config(page_title="Accent Detection App", page_icon="🎤")

st.title("🎤 Real-Time Accent Detection System")
st.write("Upload or record your voice to detect your accent.")

# -----------------------------
# Load Models (Cached)
# -----------------------------
@st.cache_resource
def load_models():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model.eval()

    clf = joblib.load("accent_classifier.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    return processor, wav2vec_model, clf, label_encoder

processor, wav2vec_model, clf, label_encoder = load_models()

# -----------------------------
# Extract Embedding
# -----------------------------
def extract_embedding(audio_path):
    waveform, sr = librosa.load(audio_path, sr=16000)

    # Normalize
    if np.max(np.abs(waveform)) > 0:
        waveform = waveform / np.max(np.abs(waveform))

    inputs = processor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = wav2vec_model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.squeeze().numpy()

# -----------------------------
# Prediction Function
# -----------------------------
def predict_accent(audio_path):
    embedding = extract_embedding(audio_path).reshape(1, -1)

    prediction = clf.predict(embedding)
    probabilities = clf.predict_proba(embedding)[0]

    predicted_label = label_encoder.inverse_transform(prediction)[0]
    confidence = float(np.max(probabilities)) * 100

    return predicted_label, confidence, probabilities

# -----------------------------
# Function to Handle Audio
# -----------------------------
def process_audio(audio_path):

    waveform, sr = librosa.load(audio_path, sr=16000)
    duration = len(waveform) / sr

    if duration < 1.0:
        st.warning("⚠️ Audio too short. Please provide at least 1 second.")
        return

    if np.abs(waveform).mean() < 0.01:
        st.warning("⚠️ No clear voice detected. Please speak clearly.")
        return

    st.audio(audio_path)

    predicted_label, confidence, probabilities = predict_accent(audio_path)

    # Show Prediction
    st.success(f"🎯 Predicted Accent: {predicted_label}")
    st.write(f"### Confidence: {round(confidence, 2)}%")

    # Show Probability Distribution
    st.write("### Accent Probability Distribution")

    for label, prob in zip(label_encoder.classes_, probabilities):
        st.write(f"{label}: {round(prob*100, 2)}%")
        st.progress(float(prob))


# -----------------------------
# Upload Section
# -----------------------------
st.subheader("📁 Upload Audio File")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    process_audio("temp.wav")
    os.remove("temp.wav")

# -----------------------------
# Live Recording Section
# -----------------------------
st.subheader("🎙️ Record Live Audio")

audio_bytes = audio_recorder()

if audio_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    process_audio(tmp_path)
    os.remove(tmp_path)