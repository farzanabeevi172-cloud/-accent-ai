import os
import librosa
import numpy as np

DATASET_PATH = "data"

X = []
y = []

label_map = {
    "indian": 0,
    "uk": 1,
    "us": 2
}

for accent in os.listdir(DATASET_PATH):
    accent_path = os.path.join(DATASET_PATH, accent)

    if os.path.isdir(accent_path):
        for file in os.listdir(accent_path):
            if file.endswith(".wav"):
                file_path = os.path.join(accent_path, file)

                print(f"Processing {file_path}")

                audio, sr = librosa.load(file_path, sr=16000)

                # Extract MFCC features
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                mfcc_mean = np.mean(mfcc.T, axis=0)

                X.append(mfcc_mean)
                y.append(label_map[accent])

X = np.array(X)
y = np.array(y)

np.save("X.npy", X)
np.save("y.npy", y)

print("Dataset prepared successfully!")