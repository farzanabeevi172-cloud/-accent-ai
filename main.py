from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import librosa
import numpy as np
import tempfile
import os
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Accent Predictor API", version="1.0.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# LOAD MODEL
# -----------------------------
try:
    model = joblib.load(os.path.join(BASE_DIR, "accent_model.pkl"))
    logger.info(f"✅ Model loaded | type: {type(model).__name__} | features: {model.n_features_in_}")
except Exception as e:
    raise RuntimeError(f"❌ accent_model.pkl failed to load: {e}")

# Matches train_model.py exactly
label_names = ["Indian", "UK", "US"]

# -----------------------------
# FEATURE EXTRACTION
# Matches predict.py exactly: 40 MFCC means
# -----------------------------
def extract_features(audio_path: str) -> np.ndarray:
    audio, sr = librosa.load(audio_path, sr=16000)

    if len(audio) < 1600:
        raise ValueError("Audio too short.")

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)  # shape: (40,) — matches X.npy

    return mfcc_mean

# -----------------------------
# PREDICT
# -----------------------------
def predict_accent(audio_path: str):
    features = extract_features(audio_path).reshape(1, -1)

    predicted_id = int(model.predict(features)[0])
    accent = label_names[predicted_id]

    probs = model.predict_proba(features)[0]
    confidence = round(float(np.max(probs)) * 100, 2)
    all_probs = {
        label_names[i]: round(float(probs[i]) * 100, 2)
        for i in range(len(label_names))
    }

    return accent, confidence, all_probs

# -----------------------------
# ROUTES
# -----------------------------
@app.get("/")
def root():
    return {
        "message": "Accent Predictor API is running.",
        "pipeline": "MFCC (40 features) → RandomForest",
        "training_samples": 49,
        "features_expected": model.n_features_in_,
        "labels": label_names
    }

@app.get("/labels")
def get_labels():
    return {"labels": label_names}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = None
    try:
        suffix = os.path.splitext(file.filename or "audio.wav")[-1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            if len(contents) == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            tmp.write(contents)
            temp_path = tmp.name

        accent, confidence, all_probs = predict_accent(temp_path)

        return JSONResponse(content={
            "predicted_accent": accent,
            "confidence_score": confidence,
            "all_probabilities": all_probs
        })

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)