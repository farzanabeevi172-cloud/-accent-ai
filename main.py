from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import tempfile
import os
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Accent Predictor API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# LOAD MODEL
# -----------------------------
try:
    model = joblib.load(os.path.join(BASE_DIR, "accent_classifier.pkl"))  # FIXED filename
    logger.info(f"✅ Model loaded | type: {type(model).__name__} | features: {model.n_features_in_}")
except Exception as e:
    logger.error(f"❌ accent_classifier.pkl failed to load: {e}")
    model = None

try:
    label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
    logger.info("✅ Label encoder loaded")
except Exception as e:
    logger.error(f"❌ label_encoder.pkl failed to load: {e}")
    label_encoder = None

try:
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    logger.info("✅ Scaler loaded")
except Exception as e:
    logger.error(f"❌ scaler.pkl failed to load: {e}")
    scaler = None

# Matches train_model.py exactly
label_names = ["Indian", "UK", "US"]

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(audio_path: str) -> np.ndarray:
    try:
        audio, sr = librosa.load(audio_path, sr=16000)

        if len(audio) < 1600:  # 0.1 seconds at 16kHz
            raise ValueError(f"Audio too short. Length: {len(audio)} samples")

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)  # shape: (40,)

        return mfcc_mean

    except Exception as e:
        logger.error(f"Feature extraction error: {str(e)}")
        raise

# -----------------------------
# PREDICT
# -----------------------------
def predict_accent(audio_path: str):
    if model is None:
        raise ValueError("Model not loaded. Please check server logs.")

    features = extract_features(audio_path).reshape(1, -1)

    # Apply scaler if loaded
    if scaler is not None:
        features = scaler.transform(features)

    # Check feature dimension matches model expectation
    if features.shape[1] != model.n_features_in_:
        raise ValueError(f"Feature dimension mismatch. Expected {model.n_features_in_}, got {features.shape[1]}")

    predicted_id = int(model.predict(features)[0])

    # Use label encoder if available, otherwise use label_names
    if label_encoder is not None:
        accent = str(label_encoder.inverse_transform([predicted_id])[0])
    else:
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
    model_status = "loaded" if model is not None else "not loaded"
    return {
        "message": "Accent Predictor API is running.",
        "model_status": model_status,
        "pipeline": "MFCC (40 features) → SVM",
        "features_expected": model.n_features_in_ if model is not None else 0,
        "labels": label_names
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }

@app.get("/labels")
def get_labels():
    return {"labels": label_names}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = None
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")

        if not file.filename or not file.filename.endswith('.wav'):
            raise HTTPException(status_code=400, detail="Only WAV files are supported")

        suffix = os.path.splitext(file.filename or "audio.wav")[-1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            if len(contents) == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")

            if len(contents) > 200 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="File too large. Maximum size is 200MB.")

            tmp.write(contents)
            temp_path = tmp.name

        logger.info(f"Processing file: {file.filename}, size: {len(contents)/1024:.1f}KB")

        accent, confidence, all_probs = predict_accent(temp_path)

        logger.info(f"Prediction: {accent} with {confidence}% confidence")

        return JSONResponse(content={
            "predicted_accent": accent,
            "confidence_score": confidence,
            "all_probabilities": all_probs
        })

    except ValueError as ve:
        logger.warning(f"ValueError: {str(ve)}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_path}: {e}")

@app.post("/test")
async def test_prediction():
    return {"message": "Test endpoint - upload a file to /predict"}