import sys
import numpy as np
import librosa
import joblib
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# -----------------------------
# Load trained files
# -----------------------------
model = joblib.load("accent_classifier.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# -----------------------------
# Load Wav2Vec2
# -----------------------------
print("Loading Wav2Vec2...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model.eval()

# -----------------------------
# Check input
# -----------------------------
if len(sys.argv) < 2:
    print("Usage: python predict.py audio.wav")
    sys.exit()

audio_path = sys.argv[1]

print(f"\nProcessing file: {audio_path}")

# -----------------------------
# Load audio
# -----------------------------
waveform, sr = librosa.load(audio_path, sr=16000)

# Normalize
if np.max(np.abs(waveform)) > 0:
    waveform = waveform / np.max(np.abs(waveform))

# -----------------------------
# Extract Wav2Vec2 embedding
# -----------------------------
inputs = processor(
    waveform,
    sampling_rate=16000,
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    outputs = wav2vec_model(**inputs)

embedding = outputs.last_hidden_state.mean(dim=1)
embedding = embedding.squeeze().numpy()

# -----------------------------
# Scale features
# -----------------------------
embedding = scaler.transform([embedding])

# -----------------------------
# Predict
# -----------------------------
prediction = model.predict(embedding)
accent = label_encoder.inverse_transform(prediction)[0]

print("\n🎙 Predicted Accent:", accent)