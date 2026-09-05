"""
backend.py - Real deepfake-voice detection API
================================================
Model  : garystafford/wav2vec2-deepfake-voice-detector
Server : uvicorn backend:app --host 0.0.0.0 --port 8000

Endpoints
---------
POST /predict   - Upload an audio file; returns label / confidence / risk_level
GET  /health    - Liveness check; confirms model is loaded

JSON contract (Flutter-compatible)
-----------------------------------
{
  "label":      "bonafide" | "spoof",
  "confidence": float  0.0-1.0,
  "risk_level": "low" | "medium" | "high"
}
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from typing import Any

import librosa
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"
SAMPLE_RATE = 16_000
FAKE_THRESHOLD = 0.4          # probs[1] >= threshold -> fake  (matches test_pretrained.py)
HIGH_RISK_THRESHOLD = 0.85    # spoof confidence for "high" risk

ALLOWED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".webm"}

# ---------------------------------------------------------------------------
# Global model state (populated in lifespan)
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {
    "model": None,
    "feature_extractor": None,
    "device": None,
    "model_loaded": False,
}


# ---------------------------------------------------------------------------
# Lifespan: load model once at startup, release at shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading model '%s' ...", MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
    model.to(device).eval()

    _state["model"] = model
    _state["feature_extractor"] = feature_extractor
    _state["device"] = device
    _state["model_loaded"] = True
    log.info("Model loaded successfully. Labels: %s", model.config.id2label)

    yield  # -- server is running --

    log.info("Shutting down - releasing model.")
    _state["model"] = None
    _state["feature_extractor"] = None
    _state["model_loaded"] = False


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Deepfake Voice Detector",
    description="Real-time audio deepfake detection powered by wav2vec2.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # fine for local dev/demo; tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run_inference(audio_path: str) -> dict:
    """
    Run the wav2vec2 deepfake classifier on a 16kHz mono audio clip.

    Returns raw model outputs:
        {
          "label": "bonafide" | "spoof",
          "confidence": float 0.0-1.0,
          "risk_level": "low" | "medium" | "high",
          "prob_real": float,
          "prob_fake": float,
        }
    """
    model: AutoModelForAudioClassification = _state["model"]
    feature_extractor: AutoFeatureExtractor = _state["feature_extractor"]
    device: str = _state["device"]

    # --- load audio -------------------------------------------------------
    try:
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot read audio file: {exc}",
        ) from exc

    if audio is None or len(audio) == 0:
        raise HTTPException(status_code=422, detail="Audio file is empty or unreadable.")

    # --- feature extraction -----------------------------------------------
    try:
        inputs = feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Feature extraction failed: {exc}",
        ) from exc

    # --- model forward pass -----------------------------------------------
    try:
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {exc}",
        ) from exc

    # probs[0] = "real" (bonafide), probs[1] = "fake" (spoof)
    prob_real: float = probs[0].item()
    prob_fake: float = probs[1].item()

    # --- decision ---------------------------------------------------------
    is_fake = prob_fake >= FAKE_THRESHOLD
    label = "spoof" if is_fake else "bonafide"
    # confidence = probability of whichever class was predicted
    confidence = prob_fake if is_fake else prob_real

    # --- risk level -------------------------------------------------------
    if label == "spoof" and confidence >= HIGH_RISK_THRESHOLD:
        risk_level = "high"
    elif label == "spoof":
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "label": label,
        "confidence": round(confidence, 6),
        "risk_level": risk_level,
        "prob_real": round(prob_real, 6),
        "prob_fake": round(prob_fake, 6),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an uploaded audio file and return a deepfake detection result.

    Returns:
        {"label": "bonafide"|"spoof", "confidence": float, "risk_level": "low"|"medium"|"high"}
    """
    if not _state["model_loaded"]:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    # --- validate file type -----------------------------------------------
    filename = file.filename or ""
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    # --- save to temp file ------------------------------------------------
    suffix = ext if ext else ".wav"
    tmp_path = os.path.join(
        tempfile.gettempdir(),
        f"deepfake_{uuid.uuid4().hex}{suffix}",
    )
    try:
        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to read uploaded file: {exc}",
        ) from exc

    # --- run inference ----------------------------------------------------
    try:
        result = _run_inference(tmp_path)
    finally:
        # always clean up temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # --- log & return -----------------------------------------------------
    log.info(
        "PREDICT | file=%s | label=%s | confidence=%.4f | risk=%s",
        filename,
        result["label"],
        result["confidence"],
        result["risk_level"],
    )

    # Flutter contract fields + DEBUG raw probs (remove before final demo build)
    return {
        "label": result["label"],
        "confidence": result["confidence"],
        "risk_level": result["risk_level"],
        "debug_prob_real": result["prob_real"],   # DEBUG — strip before demo
        "debug_prob_fake": result["prob_fake"],   # DEBUG — strip before demo
    }


@app.get("/health")
async def health():
    """Liveness / readiness check."""
    return {"status": "ok", "model_loaded": _state["model_loaded"]}
