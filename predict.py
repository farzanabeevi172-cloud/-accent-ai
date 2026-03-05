import sys
import numpy as np
import librosa
import joblib

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("accent_model.pkl")

# Label mapping (same order as training)
label_names = ["Indian", "UK", "US"]

# -----------------------------
# Check input file
# -----------------------------
if len(sys.argv) < 2:
    print("Usage: python predict.py path_to_audio.wav")
    sys.exit()

audio_path = sys.argv[1]

print(f"\nProcessing file: {audio_path}")

# -----------------------------
# Load audio (16kHz)
# -----------------------------
audio, sr = librosa.load(audio_path, sr=16000)

# -----------------------------
# Extract MFCC
# -----------------------------
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
mfcc_mean = np.mean(mfcc.T, axis=0)

# Reshape for model
features = mfcc_mean.reshape(1, -1)

# -----------------------------
# Predict
# -----------------------------
prediction = model.predict(features)[0]
predicted_accent = label_names[prediction]

print("\n🎙 Predicted Accent:", predicted_accent)