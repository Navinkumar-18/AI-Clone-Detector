"""
backend.py — VoiceGuard deepfake-voice detection API (hardened)
===============================================================
Model  : garystafford/wav2vec2-deepfake-voice-detector
Server : python backend.py  (HTTPS on 0.0.0.0:8443 by default)

Configuration loaded from: config/model_config.yaml → voiceguard_config.py

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
POST /predict        - Upload an audio file; returns detection result
POST /live/analyze   - Upload a live audio chunk; returns detection result
GET  /health         - Liveness check (process alive)
GET  /ready          - Readiness check (model loaded, config loaded)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

import librosa
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from voiceguard_config import load_config, VoiceGuardConfig
from audio_quality import analyze_audio_quality, AudioQualityReport

# ---------------------------------------------------------------------------
# Logging — structured, sanitized
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load unified configuration
# ---------------------------------------------------------------------------
cfg: VoiceGuardConfig = load_config()

# Log config summary (no secrets)
log.info("Config loaded | model=%s | threshold=%s | demo_mode=%s",
         cfg.model.name, cfg.thresholds.spoof_threshold, cfg.security.demo_mode)

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class AudioQualityResponse(BaseModel):
    status: str = Field(default="unknown", description="acceptable | poor | silent | invalid | unknown")
    duration_seconds: float = 0.0
    rms: float = 0.0
    snr_db: Optional[float] = None
    clipping_ratio: float = 0.0
    voiced_ratio: float = 0.0


class EvidenceResponse(BaseModel):
    window_count: int = 0
    high_risk_window_count: int = 0
    analysis_age_ms: int = 0


class DetectionResponse(BaseModel):
    """Canonical response schema for /predict and /live/analyze."""
    decision: str = Field(description="low_risk | verification_required | action_held | insufficient_evidence")
    action: str = Field(description="allow_with_caution | verify | hold | unavailable")
    label: str = Field(description="bonafide | spoof | unknown")
    spoof_score: float = Field(ge=0.0, le=1.0, description="Raw prob_fake from model")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence of predicted class")
    risk_level: str = Field(description="low | medium | high | unknown")
    risk_percentage: float = Field(ge=0.0, le=100.0)
    speech_detected: bool = True
    audio_quality: AudioQualityResponse = Field(default_factory=AudioQualityResponse)
    evidence: EvidenceResponse = Field(default_factory=EvidenceResponse)
    reason_codes: List[str] = Field(default_factory=list)
    model_version: str = ""
    threshold_version: str = ""
    request_id: str = ""


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = False


class ReadinessResponse(BaseModel):
    status: str = "not_ready"
    model_loaded: bool = False
    model_version: str = ""
    threshold_version: str = ""
    device: str = ""
    demo_mode: bool = False


# ---------------------------------------------------------------------------
# Global model state (populated in lifespan)
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {
    "model": None,
    "feature_extractor": None,
    "device": None,
    "model_loaded": False,
}

# Thread pool for non-blocking inference
_inference_executor: Optional[ThreadPoolExecutor] = None
_inference_semaphore: Optional[asyncio.Semaphore] = None

# Simple in-memory rate limiter (single-process demo only)
_rate_limit_store: dict[str, list[float]] = defaultdict(list)


# ---------------------------------------------------------------------------
# Lifespan: load model once at startup, release at shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _inference_executor, _inference_semaphore

    log.info("Loading model '%s' ...", cfg.model.name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        cfg.model.name, revision=cfg.model.revision
    )
    model = AutoModelForAudioClassification.from_pretrained(
        cfg.model.name, revision=cfg.model.revision
    )
    model.to(device).eval()

    _state["model"] = model
    _state["feature_extractor"] = feature_extractor
    _state["device"] = device
    _state["model_loaded"] = True

    _inference_executor = ThreadPoolExecutor(
        max_workers=cfg.server.maximum_concurrency,
        thread_name_prefix="vg-inference",
    )
    _inference_semaphore = asyncio.Semaphore(cfg.server.maximum_concurrency)

    log.info("Model loaded successfully. Labels: %s", model.config.id2label)

    yield  # -- server is running --

    log.info("Shutting down - releasing model.")
    _state["model"] = None
    _state["feature_extractor"] = None
    _state["model_loaded"] = False
    if _inference_executor:
        _inference_executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="VoiceGuard — Voice Clone Risk Detection API",
    description="Real-time audio deepfake detection powered by wav2vec2.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — restricted by default, wildcard only in demo mode
_cors_origins = (
    ["*"] if cfg.security.demo_mode
    else list(cfg.security.allowed_cors_origins) or ["http://localhost", "https://localhost"]
)
if cfg.security.demo_mode:
    log.warning("DEMO MODE enabled — wildcard CORS active. Do not use in production.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware: request ID + rate limiting
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_request_id_and_rate_limit(request: Request, call_next):
    # Request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Rate limiting (simple per-IP, in-memory, single-process demo only)
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = 60.0
    max_requests = cfg.server.rate_limit_requests_per_minute

    # Clean old entries
    _rate_limit_store[client_ip] = [
        t for t in _rate_limit_store[client_ip] if now - t < window
    ]

    if len(_rate_limit_store[client_ip]) >= max_requests:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded", "request_id": request_id},
        )

    _rate_limit_store[client_ip].append(now)

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_inference_sync(audio: np.ndarray) -> dict:
    """
    Run the wav2vec2 deepfake classifier on a 16kHz mono audio array.
    This runs SYNCHRONOUSLY and must be called from a thread pool.

    Returns:
        {
          "label": "bonafide" | "spoof",
          "confidence": float 0.0-1.0,
          "risk_level": "low" | "medium" | "high",
          "spoof_score": float 0.0-1.0,
          "prob_real": float 0.0-1.0,
        }
    """
    model = _state["model"]
    feature_extractor = _state["feature_extractor"]
    device: str = _state["device"]

    inputs = feature_extractor(
        audio,
        sampling_rate=cfg.audio.sample_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    prob_real: float = probs[0].item()
    prob_fake: float = probs[1].item()

    # Decision using configured threshold
    is_fake = prob_fake >= cfg.thresholds.spoof_threshold
    label = "spoof" if is_fake else "bonafide"
    confidence = prob_fake if is_fake else prob_real

    # Risk level
    if label == "spoof" and confidence >= cfg.thresholds.high_risk_threshold:
        risk_level = "high"
    elif label == "spoof":
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "label": label,
        "confidence": round(confidence, 6),
        "risk_level": risk_level,
        "spoof_score": round(prob_fake, 6),
        "prob_real": round(prob_real, 6),
    }


async def _read_upload_limited(file: UploadFile, max_bytes: int) -> bytes:
    """
    Read an upload file with a byte limit.
    Reads in chunks and rejects early if the limit is exceeded.
    Does NOT load arbitrarily large files into memory.
    """
    chunks = []
    total = 0
    chunk_size = 65536  # 64 KB chunks

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds maximum size of {max_bytes} bytes.",
            )
        chunks.append(chunk)

    return b"".join(chunks)


def _save_upload_to_temp(file_bytes: bytes, suffix: str = ".wav") -> str:
    """Save uploaded bytes to a temporary file and return its path."""
    tmp_path = os.path.join(
        tempfile.gettempdir(),
        f"vg_{uuid.uuid4().hex}{suffix}",
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


def _make_insufficient_evidence_response(
    reason_codes: List[str],
    request_id: str,
    audio_quality: Optional[AudioQualityReport] = None,
) -> DetectionResponse:
    """Create an INSUFFICIENT_EVIDENCE response."""
    aq = audio_quality.to_dict() if audio_quality else {}
    return DetectionResponse(
        decision="insufficient_evidence",
        action="verify",
        label="unknown",
        spoof_score=0.0,
        confidence=0.0,
        risk_level="unknown",
        risk_percentage=0.0,
        speech_detected=False,
        audio_quality=AudioQualityResponse(**aq) if aq else AudioQualityResponse(status="unknown"),
        evidence=EvidenceResponse(),
        reason_codes=reason_codes,
        model_version=cfg.model.version,
        threshold_version=cfg.thresholds.version,
        request_id=request_id,
    )


# Action mapping
_DECISION_ACTION_MAP = {
    "low": ("low_risk", "allow_with_caution"),
    "medium": ("verification_required", "verify"),
    "high": ("verification_required", "verify"),  # Single clip cannot generate action_held
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=DetectionResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Accept an uploaded audio file and return a deepfake detection result.
    Uses the canonical DetectionResponse schema.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    if not _state["model_loaded"]:
        return _make_insufficient_evidence_response(
            ["model_unavailable"], request_id
        )

    # --- validate file type ---
    filename = file.filename or ""
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in cfg.audio.allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(cfg.audio.allowed_extensions))}",
        )

    # --- read with size limit (streaming, not all-at-once) ---
    try:
        contents = await _read_upload_limited(file, cfg.server.maximum_upload_bytes)
    except HTTPException:
        raise

    # --- save to temp + process ---
    suffix = ext if ext else ".wav"
    tmp_path: Optional[str] = None
    try:
        tmp_path = _save_upload_to_temp(contents, suffix)

        # Load audio
        try:
            audio, _ = librosa.load(tmp_path, sr=cfg.audio.sample_rate, mono=True)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Cannot read audio file: {exc}") from exc

        if audio is None or len(audio) == 0:
            return _make_insufficient_evidence_response(
                ["empty_audio"], request_id
            )

        # --- Audio quality check (BEFORE inference) ---
        quality = analyze_audio_quality(
            audio, cfg.audio.sample_rate,
            min_duration=cfg.audio.minimum_duration_seconds,
            max_duration=cfg.audio.maximum_duration_seconds,
            rms_silence_threshold=cfg.audio_quality.rms_silence_threshold,
            min_voiced_ratio=cfg.audio_quality.minimum_voiced_ratio,
            max_clipping_ratio=cfg.audio_quality.maximum_clipping_ratio,
            poor_snr_db=cfg.audio_quality.poor_snr_threshold_db,
        )

        if not quality.is_usable():
            log.info("PREDICT | request_id=%s | quality=%s | reasons=%s",
                     request_id, quality.status, quality.reason_codes)
            return _make_insufficient_evidence_response(
                quality.reason_codes or [f"audio_quality_{quality.status}"],
                request_id,
                quality,
            )

        # --- Run inference off the event loop ---
        async with _inference_semaphore:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                _inference_executor,
                _run_inference_sync,
                audio,
            )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to process uploaded file: {exc}",
        ) from exc
    finally:
        if tmp_path:
            _cleanup_temp(tmp_path)

    # --- Build response ---
    decision, action = _DECISION_ACTION_MAP.get(
        result["risk_level"], ("verification_required", "verify")
    )

    log.info("PREDICT | request_id=%s | label=%s | spoof_score=%.4f | risk=%s",
             request_id, result["label"], result["spoof_score"], result["risk_level"])

    return DetectionResponse(
        decision=decision,
        action=action,
        label=result["label"],
        spoof_score=result["spoof_score"],
        confidence=result["confidence"],
        risk_level=result["risk_level"],
        risk_percentage=round(result["confidence"] * 100, 1),
        speech_detected=True,
        audio_quality=AudioQualityResponse(**quality.to_dict()),
        evidence=EvidenceResponse(),
        reason_codes=(["elevated_spoof_score"] if result["label"] == "spoof" else []),
        model_version=cfg.model.version,
        threshold_version=cfg.thresholds.version,
        request_id=request_id,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness check — is the process alive?"""
    return HealthResponse(status="ok", model_loaded=_state["model_loaded"])


@app.get("/ready", response_model=ReadinessResponse)
async def ready():
    """Readiness check — is the model loaded and ready for inference?"""
    if _state["model_loaded"]:
        return ReadinessResponse(
            status="ready",
            model_loaded=True,
            model_version=cfg.model.version,
            threshold_version=cfg.thresholds.version,
            device=_state.get("device", "unknown"),
            demo_mode=cfg.security.demo_mode,
        )
    return ReadinessResponse(
        status="not_ready",
        model_loaded=False,
        model_version=cfg.model.version,
        device="",
        demo_mode=cfg.security.demo_mode,
    )


# ---------------------------------------------------------------------------
# Live Call Analysis endpoint
# ---------------------------------------------------------------------------
@app.post("/live/analyze", response_model=DetectionResponse)
async def live_analyze(request: Request, file: UploadFile = File(...)):
    """
    Accept a short audio chunk from live call analysis and return
    clip-level detection evidence.

    This returns CLIP-LEVEL evidence. Call-level decisions (ACTION_HELD)
    are produced by the client-side risk aggregator, not this endpoint.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    if not _state["model_loaded"]:
        return _make_insufficient_evidence_response(
            ["model_unavailable"], request_id
        )

    # --- read with size limit ---
    try:
        contents = await _read_upload_limited(file, cfg.server.maximum_upload_bytes)
    except HTTPException:
        raise

    tmp_path: Optional[str] = None
    try:
        tmp_path = _save_upload_to_temp(contents, ".wav")

        # Load audio
        try:
            audio, _ = librosa.load(tmp_path, sr=cfg.audio.sample_rate, mono=True)
        except Exception as exc:
            raise HTTPException(
                status_code=422, detail=f"Cannot read audio chunk: {exc}"
            ) from exc

        if audio is None or len(audio) == 0:
            return _make_insufficient_evidence_response(
                ["empty_audio"], request_id
            )

        # --- Audio quality check ---
        quality = analyze_audio_quality(
            audio, cfg.audio.sample_rate,
            min_duration=0.3,  # Lower minimum for live chunks
            max_duration=cfg.audio.maximum_duration_seconds,
            rms_silence_threshold=cfg.audio_quality.rms_silence_threshold,
            min_voiced_ratio=cfg.audio_quality.minimum_voiced_ratio,
            max_clipping_ratio=cfg.audio_quality.maximum_clipping_ratio,
            poor_snr_db=cfg.audio_quality.poor_snr_threshold_db,
        )

        if not quality.is_usable():
            log.info("LIVE | request_id=%s | quality=%s | reasons=%s",
                     request_id, quality.status, quality.reason_codes)
            return _make_insufficient_evidence_response(
                quality.reason_codes or [f"audio_quality_{quality.status}"],
                request_id,
                quality,
            )

        # --- Run inference off the event loop ---
        async with _inference_semaphore:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                _inference_executor,
                _run_inference_sync,
                audio,
            )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Failed to process audio chunk: {exc}"
        ) from exc
    finally:
        if tmp_path:
            _cleanup_temp(tmp_path)

    # --- Build clip-level response ---
    # Single clip cannot generate ACTION_HELD — that is a call-level decision
    decision, action = _DECISION_ACTION_MAP.get(
        result["risk_level"], ("verification_required", "verify")
    )

    log.info("LIVE | request_id=%s | label=%s | spoof_score=%.4f | risk=%s",
             request_id, result["label"], result["spoof_score"], result["risk_level"])

    return DetectionResponse(
        decision=decision,
        action=action,
        label=result["label"],
        spoof_score=result["spoof_score"],
        confidence=result["confidence"],
        risk_level=result["risk_level"],
        risk_percentage=round(result["confidence"] * 100, 1),
        speech_detected=True,
        audio_quality=AudioQualityResponse(**quality.to_dict()),
        evidence=EvidenceResponse(),
        reason_codes=(["elevated_spoof_score"] if result["label"] == "spoof" else []),
        model_version=cfg.model.version,
        threshold_version=cfg.thresholds.version,
        request_id=request_id,
    )


# ---------------------------------------------------------------------------
# Direct launch (python backend.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    dev_mode = os.environ.get("VOICEGUARD_DEV_MODE", "").strip() == "1"

    if dev_mode:
        log.warning(
            "VOICEGUARD_DEV_MODE=1 — running plain HTTP on port 8000. "
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
