import os
import numpy as np
import librosa
from pathlib import Path

DATA_DIR = "data"
LABELS = {"indian": 0, "uk": 1, "us": 2}

X, y = [], []

for label_name, label_id in LABELS.items():
    folder = Path(DATA_DIR) / label_name
    files = list(folder.glob("*.wav"))
    print(f"{label_name}: {len(files)} files found")
    for f in files:
        try:
            audio, sr = librosa.load(str(f), sr=16000)
            if len(audio) < 1600:
                print(f"Skipping {f}: too short")
                continue
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfcc_mean = np.mean(mfcc.T, axis=0)
            X.append(mfcc_mean)
            y.append(label_id)
        except Exception as e:
            print(f"Skipping {f}: {e}")

X = np.array(X)
y = np.array(y)
np.save("X.npy", X)
np.save("y.npy", y)
print(f"Done! Saved X.npy {X.shape} and y.npy {y.shape}")