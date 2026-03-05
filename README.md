# Accent AI – 3‑Accent Detection Web App

Production-ready pipeline and web app for detecting **Indian**, **American**, and **British** English accents using **Mozilla Common Voice** and **wav2vec2-base**.

---

## 1. Environment Setup

- **Python**: 3.9–3.11 recommended.
- **Node**: 18+ recommended for the frontend.

Install Python dependencies:

```bash
cd accent-ai
pip install -r requirements.txt
```

> If you’re on Google Colab, upload this folder, `cd` into it, and run the same `pip install` command.

---

## 2. Download & Preprocess Common Voice

The preprocessing script uses the Hugging Face **datasets** interface for Mozilla Common Voice.

### 2.1. Hugging Face / Common Voice Access

- You may need to accept the Common Voice terms on Hugging Face first.
- Visit: `https://huggingface.co/datasets/mozilla-foundation/common_voice_16_0`
- Click **“Access repository”** / accept any terms if prompted.

### 2.2. Run Preprocessing

This will:
- Load `mozilla-foundation/common_voice_16_0`, config `en`.
- Filter to **Indian (en-IN)**, **American (en-US)**, **British (en-GB)**.
- Enforce **≥ 3 seconds** audio, **16kHz mono**.
- Balance classes and create an **80/20 train/val** split.

```bash
python data_preprocessing.py \
  --dataset_name mozilla-foundation/common_voice_16_0 \
  --dataset_config en \
  --output_dir data \
  --min_duration 3.0 \
  --max_per_class 2000
```

Outputs:
- `data/audio/train/*.wav`
- `data/audio/val/*.wav`
- `data/metadata_train.csv`
- `data/metadata_val.csv`
- `data/label_mapping.json`

---

## 3. Train wav2vec2 Accent Classifier

Fine-tune **facebook/wav2vec2-base** for 3‑class accent classification.

```bash
python train.py \
  --data_dir data \
  --output_dir model \
  --model_name facebook/wav2vec2-base \
  --num_epochs 8 \
  --batch_size 8 \
  --lr 1e-5
```

Key details:
- Uses **CrossEntropyLoss** with **class weights** (to handle imbalance).
- Tracks **accuracy** and **macro F1**.
- Uses a **linear LR scheduler with warmup**.
- Implements **early stopping** on validation F1.
- Saves the best model + processor to:
  - `model/accent_model/`

> For Colab: enable GPU (`Runtime → Change runtime type → GPU`) to speed up training.

---

## 4. Run FastAPI Backend

The backend:
- Loads the trained model from `model/accent_model`.
- Exposes `POST /predict` and `GET /health`.
- Performs **silence / no-voice detection** using energy-based VAD (librosa).

Start the API:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or:

```bash
python main.py
```

### 4.1. API Contract

- **Endpoint**: `POST /predict`
- **Body**: `multipart/form-data` with a single file field named `file` (audio file).

Responses:

- If no speech detected:

```json
{
  "status": "no_voice",
  "message": "No voice detected"
}
```

- If speech is detected:

```json
{
  "accent": "Indian",
  "confidence": 0.92
}
```

Health check:

- `GET /health` → `{ "status": "ok", "model_loaded": true/false }`

---

## 5. Run React Frontend

The frontend lives in `frontend/` and is built with **React + Vite**.

From `accent-ai/frontend`:

```bash
npm install
npm run dev
```

Vite dev server defaults to `http://localhost:5173`.

### 5.1. Frontend–Backend Integration

- Frontend calls:
  - `POST http://localhost:8000/predict`
- Make sure the backend is running **before** you click “Start Recording”.
- If you deploy backend/frontend separately, update `API_URL` in `frontend/src/App.jsx`.

### 5.2. UI Features

- **Blue–purple gradient** background with glassmorphism card.
- **Microphone record button**:
  - Uses `MediaRecorder` to capture microphone audio.
  - Shows a **wave animation** while recording.
- On stop:
  - Sends audio blob as `file` to `/predict`.
  - Displays:
    - Detected accent.
    - Confidence percentage (0–100%).
    - Or **red “No Voice Detected”** message if backend returns that status.

---

## 6. Audio / ffmpeg Notes

- `librosa` can load **WAV** out of the box.
- For **webm/opus** (MediaRecorder default in many browsers), `librosa` typically relies on `ffmpeg` or `av`.
- If you see errors like **“No backend available to read the file”**:
  - Install ffmpeg and ensure it’s on your `PATH`.

### 6.1. Install ffmpeg (examples)

**Windows (choco):**

```bash
choco install ffmpeg
```

**macOS (Homebrew):**

```bash
brew install ffmpeg
```

**Ubuntu / Debian:**

```bash
sudo apt update
sudo apt install ffmpeg
```

Restart your shell / IDE after installation so `ffmpeg` is picked up.

---

## 7. Troubleshooting

- **Model not loaded / 500 on /predict**:
  - Ensure you ran `train.py` and that `model/accent_model/` exists.
  - Check `/health` to see `model_loaded: true`.

- **No voice detected too often**:
  - Ensure you speak at a reasonable volume near the mic.
  - You can tweak VAD thresholds in `detect_voice` (in `main.py`).

- **React app cannot reach backend**:
  - Confirm backend is running on `http://localhost:8000`.
  - Check browser console / network tab for CORS or connection errors.

- **Out of memory during training**:
  - Reduce `--batch_size`.
  - Optionally reduce `--max_per_class` in `data_preprocessing.py`.

---

## 8. Colab Quickstart (High Level)

1. Upload this `accent-ai/` folder to Colab environment (or clone from your repo).
2. `cd accent-ai`
3. `pip install -r requirements.txt`
4. Run:
   - `python data_preprocessing.py --output_dir data`
   - `python train.py --data_dir data --output_dir model`
5. Download `model/accent_model` to your local machine/server to use with the FastAPI backend.

You can then run the backend and frontend locally with your trained model.

