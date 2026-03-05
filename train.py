import os
import numpy as np
import torch
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from transformers import Wav2Vec2Processor, Wav2Vec2Model

# -----------------------------
# Dataset Path
# -----------------------------
DATASET_PATH = "data"

# -----------------------------
# Load Wav2Vec2
# -----------------------------
print("Loading Wav2Vec2 model...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model.eval()

# -----------------------------
# Feature Extraction Function
# -----------------------------
def extract_embedding(audio_path):
    waveform, sr = librosa.load(audio_path, sr=16000)

    # Normalize audio
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
# Load Dataset
# -----------------------------
X = []
y = []

print("Extracting features...")

for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)

    if os.path.isdir(label_path):
        for file in os.listdir(label_path):
            if file.endswith(".wav"):
                file_path = os.path.join(label_path, file)
                try:
                    embedding = extract_embedding(file_path)
                    X.append(embedding)
                    y.append(label)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

X = np.array(X)
y = np.array(y)

print("Feature extraction complete.")

# -----------------------------
# Encode Labels
# -----------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# -----------------------------
# Standardize Features (IMPORTANT)
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Training Balanced SVM...")

# -----------------------------
# Balanced SVM
# -----------------------------
clf = SVC(
    probability=True,
    kernel='rbf',
    class_weight='balanced'   # Prevent bias toward majority class
)

clf.fit(X_train, y_train)

# -----------------------------
# Evaluate Model
# -----------------------------
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))

# -----------------------------
# Save Everything
# -----------------------------
joblib.dump(clf, "accent_classifier.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel, Label Encoder, and Scaler saved successfully!")