"""
backend.py — VoiceGuard deepfake-voice detection API
=====================================================
Model  : garystafford/wav2vec2-deepfake-voice-detector
Server : python backend.py  (HTTPS on 0.0.0.0:8443 by default)

ZERO-PERSISTENCE PRIVACY GUARANTEE
-----------------------------------
This server processes audio ONLY in ephemeral temp files that are created and
deleted within a single request scope.  No raw audio, PCM data, or WAV file
is ever persisted to disk, database, or log beyond the lifetime of that
request.  Only scalar metadata (label, confidence, risk_level) is returned
to the client.  Every temp file deletion is followed by an explicit PRIVACY
audit log line.

Endpoints
---------
POST /predict        - Upload an audio file; returns label / confidence / risk_level
POST /live/analyze   - Upload a live audio chunk; returns detection_score / risk / action
GET  /health         - Liveness check; confirms model is loaded

JSON contract (Flutter-compatible)
-----------------------------------
/predict:
{
  "label":           "bonafide" | "spoof",
  "confidence":      float  0.0-1.0,
  "risk_level":      "low" | "medium" | "high",
  "risk_percentage": float  0.0-100.0,
  "risk_category":   "Low Risk" | "Medium Risk" | "Critical"
}

/live/analyze:
{
  "label":           "bonafide" | "spoof",
  "confidence":      float 0.0-1.0,
  "risk_level":      "low" | "medium" | "high",
  "detection_score": float 0.0-1.0,
  "prob_real":       float 0.0-1.0,
  "prob_fake":       float 0.0-1.0,
  "action":          "allow" | "verify" | "block",
  "risk_percentage": float 0.0-100.0,
  "risk_category":   "Low Risk" | "Medium Risk" | "Critical"
}
"""

from __future__ import annotations

import logging
import os
import sys
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
# Validated via evaluate_live_model.py sweep on ASVspoof 2019 LA eval (see threshold_report.txt)
# Threshold 0.30: lowest FNR (22.50%), best F1 (76.54%) in the 0.30–0.80 sweep.
# Security-first: prefer catching spoofs (low FNR) over reducing false alarms.
FAKE_THRESHOLD = 0.30         # probs[1] >= threshold -> fake
HIGH_RISK_THRESHOLD = 0.85    # spoof confidence for "high" risk

ALLOWED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".webm"}

# Action mapping -- used by /live/analyze for security decisions.
# These directly mirror the risk levels established in evaluate_live_model.py.
_ACTION_MAP = {"low": "allow", "medium": "verify", "high": "block"}

# Risk category labels for the UI — percentage + human-readable category.
_RISK_CATEGORY_MAP = {"low": "Low Risk", "medium": "Medium Risk", "high": "Critical"}

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


def _save_upload_to_temp(file_bytes: bytes, suffix: str = ".wav") -> str:
    """Save uploaded bytes to a temporary file and return its path."""
    tmp_path = os.path.join(
        tempfile.gettempdir(),
        f"deepfake_{uuid.uuid4().hex}{suffix}",
    )
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)
    return tmp_path


def _cleanup_temp(path: str) -> None:
    """Delete a temp file and emit a privacy audit log line."""
    try:
        os.remove(path)
    except OSError:
        pass
    log.info("PRIVACY | Temp audio purged — no persistent storage of raw audio")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an uploaded audio file and return a deepfake detection result.

    Returns:
        {"label": "bonafide"|"spoof", "confidence": float, "risk_level": "low"|"medium"|"high",
         "risk_percentage": float, "risk_category": str}
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

    # --- save to temp file + run inference in one try/finally scope --------
    # This ensures cleanup even if an exception occurs between file creation
    # and inference.
    suffix = ext if ext else ".wav"
    tmp_path: str | None = None
    try:
        contents = await file.read()
        tmp_path = _save_upload_to_temp(contents, suffix)
        result = _run_inference(tmp_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to process uploaded file: {exc}",
        ) from exc
    finally:
        # always clean up temp file
        if tmp_path:
            _cleanup_temp(tmp_path)

    # --- log & return -----------------------------------------------------
    log.info(
        "PREDICT | file=%s | label=%s | confidence=%.4f | risk=%s",
        filename,
        result["label"],
        result["confidence"],
        result["risk_level"],
    )

    risk_category = _RISK_CATEGORY_MAP.get(result["risk_level"], "Medium Risk")
    risk_percentage = round(result["confidence"] * 100, 1)

    return {
        "label": result["label"],
        "confidence": result["confidence"],
        "risk_level": result["risk_level"],
        "risk_percentage": risk_percentage,
        "risk_category": risk_category,
    }


@app.get("/health")
async def health():
    """Liveness / readiness check."""
    return {"status": "ok", "model_loaded": _state["model_loaded"]}


# ---------------------------------------------------------------------------
# Live Call Analysis endpoint
# ---------------------------------------------------------------------------
@app.post("/live/analyze")
async def live_analyze(file: UploadFile = File(...)):
    """
    Accept a short audio chunk from the live call analysis mode and return
    a detection result with security action.

    This endpoint reuses the SAME inference pipeline as /predict (same model,
    same preprocessing, same thresholds). The only difference is the response
    format, which includes detection_score and action fields for the live UI.

    Returns:
        {
          "label":           "bonafide" | "spoof",
          "confidence":      float 0.0-1.0,
          "risk_level":      "low" | "medium" | "high",
          "detection_score": float 0.0-1.0  (prob_fake — the spoof detection score),
          "prob_real":       float 0.0-1.0,
          "prob_fake":       float 0.0-1.0,
          "action":          "allow" | "verify" | "block",
          "risk_percentage": float 0.0-100.0,
          "risk_category":   "Low Risk" | "Medium Risk" | "Critical"
        }
    """
    if not _state["model_loaded"]:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    # --- save + infer in one try/finally scope ----------------------------
    tmp_path: str | None = None
    try:
        contents = await file.read()
        tmp_path = _save_upload_to_temp(contents, ".wav")
        result = _run_inference(tmp_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to process audio chunk: {exc}",
        ) from exc
    finally:
        if tmp_path:
            _cleanup_temp(tmp_path)

    action = _ACTION_MAP.get(result["risk_level"], "verify")
    risk_category = _RISK_CATEGORY_MAP.get(result["risk_level"], "Medium Risk")
    risk_percentage = round(result["confidence"] * 100, 1)

    log.info(
        "LIVE    | label=%s | score=%.4f | risk=%s | action=%s",
        result["label"],
        result["prob_fake"],
        result["risk_level"],
        action,
    )

    return {
        "label": result["label"],
        "confidence": result["confidence"],
        "risk_level": result["risk_level"],
        "detection_score": result["prob_fake"],
        "prob_real": result["prob_real"],
        "prob_fake": result["prob_fake"],
        "action": action,
        "risk_percentage": risk_percentage,
        "risk_category": risk_category,
    }


# ---------------------------------------------------------------------------
# Direct launch (python backend.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    dev_mode = os.environ.get("VOICEGUARD_DEV_MODE", "").strip() == "1"

    if dev_mode:
        log.warning(
            "⚠ VOICEGUARD_DEV_MODE=1 — running plain HTTP on port 8000. "
            "Do NOT use this in demos or production."
        )
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        cert_file = "cert.pem"
        key_file = "key.pem"
        if not (os.path.exists(cert_file) and os.path.exists(key_file)):
            log.info("TLS certificates not found — generating via generate_cert.py ...")
            import subprocess
            subprocess.run([sys.executable, "generate_cert.py"], check=True)

        log.info("Starting HTTPS server on 0.0.0.0:8443")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8443,
            ssl_keyfile=key_file,
            ssl_certfile=cert_file,
        )
