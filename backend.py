"""
backend.py — VoiceGuard deepfake-voice detection API (hardened)
===============================================================
Model  : garystafford/wav2vec2-deepfake-voice-detector
Server : python backend.py  (HTTPS on 0.0.0.0:8443 by default)

Configuration loaded from: config/model_config.yaml → voiceguard_config.py

EPHEMERAL AUDIO HANDLING
------------------------
VoiceGuard is designed for ephemeral audio processing. Temporary audio is
processed for inference and cleaned up after processing. Structured cleanup
events are logged without intentionally storing raw audio content.
Only scalar metadata (label, confidence, risk_level) is returned
to the client. Operating-system memory, swap files, crash dumps, client
buffers, infrastructure logs, and backups are outside the guarantees of
this prototype.

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
import json
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
from voiceguard.model_backends import create_backend, ModelBackend, ModelLoadError
from audio_quality import analyze_audio_quality, AudioQualityReport

# ---------------------------------------------------------------------------
# Logging — structured, sanitized (file + console)
# ---------------------------------------------------------------------------
_log_format = "%(asctime)s | %(levelname)s | %(message)s"
_date_format = "%Y-%m-%dT%H:%M:%S"

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Avoid duplicate handlers on reload
_log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend.log")
if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
    _file_handler = logging.FileHandler(_log_file_path, encoding="utf-8")
    _file_handler.setFormatter(logging.Formatter(_log_format, _date_format))
    root_logger.addHandler(_file_handler)

if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root_logger.handlers):
    _stream_handler = logging.StreamHandler(sys.stderr)
    _stream_handler.setFormatter(logging.Formatter(_log_format, _date_format))
    root_logger.addHandler(_stream_handler)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load unified configuration
# ---------------------------------------------------------------------------
cfg: VoiceGuardConfig = load_config()

# Log config summary (no secrets)
log.info("Config loaded | active_model=%s | model=%s | threshold=%s | demo_mode=%s",
         cfg.active_model, cfg.model.name, cfg.thresholds.spoof_threshold, cfg.security.demo_mode)

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
    model_backend: str = ""
    score_type: str = ""
    threshold_version: str = ""
    model_loaded: bool = Field(default=True, description="True if inference model was actively loaded")
    request_id: str = ""


class SlidingWindowItemResponse(BaseModel):
    window_index: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    window_complete: bool

    decision: str
    action: str
    label: str
    spoof_score: Optional[float] = None
    confidence: Optional[float] = None
    score_type: str = ""

    risk_level: str
    speech_detected: bool = True
    quality_status: str = "acceptable"

    rms: Optional[float] = None
    snr_db: Optional[float] = None
    voiced_ratio: Optional[float] = None
    reason_codes: List[str] = Field(default_factory=list)

    model_backend: str = ""
    model_version: str = ""
    threshold_version: str = ""
    model_loaded: bool = False


class FileSlidingAnalysisResponse(BaseModel):
    total_duration_seconds: float
    window_length_seconds: float = 4.0
    stride_seconds: float = 2.0
    window_count: int
    complete_window_count: int

    high_risk_windows: int
    maximum_spoof_score: Optional[float] = None
    average_spoof_score: Optional[float] = None
    ema_score: Optional[float] = None

    decision: str
    action: str
    persistence_triggered: bool = False
    consecutive_high_count: int = 0
    reason_codes: List[str] = Field(default_factory=list)

    model_backend: str = ""
    model_version: str = ""
    score_type: str = ""
    threshold_version: str = ""
    model_loaded: bool = False

    windows: List[SlidingWindowItemResponse] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = False


class ReadinessResponse(BaseModel):
    status: str = "not_ready"
    model_loaded: bool = False
    model_backend: str = ""
    score_type: str = ""
    model_version: str = ""
    threshold_version: str = ""
    device: str = ""
    demo_mode: bool = False


# ---------------------------------------------------------------------------
# Global model state (populated in lifespan)
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {
    "backend": None,
    "model": None,
    "feature_extractor": None,
    "device": None,
    "model_loaded": False,
    "load_error": None,
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

    log.info("Loading active model backend '%s' ...", cfg.active_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _state["device"] = device

    backend_cfg = cfg.active_backend
    try:
        model_backend = create_backend(
            backend_name=backend_cfg.backend,
            model_id=backend_cfg.model_id,
            revision=backend_cfg.revision,
            device=device,
            score_type=backend_cfg.score_type,
            checkpoint_path=backend_cfg.checkpoint_path,
        )
        model_backend.load()
        _state["backend"] = model_backend
        _state["model"] = getattr(model_backend, "model", getattr(model_backend, "backbone", None))
        _state["feature_extractor"] = getattr(model_backend, "feature_extractor", None)
        _state["model_loaded"] = True
        _state["load_error"] = None
        log.info("Model backend '%s' loaded successfully.", backend_cfg.backend)
    except Exception as exc:
        _state["backend"] = None
        _state["model"] = None
        _state["feature_extractor"] = None
        _state["model_loaded"] = False
        _state["load_error"] = str(exc)
        log.error("Failed to load active model backend '%s': %s", backend_cfg.backend, exc)

    _inference_executor = ThreadPoolExecutor(
        max_workers=cfg.server.maximum_concurrency,
        thread_name_prefix="vg-inference",
    )
    _inference_semaphore = asyncio.Semaphore(cfg.server.maximum_concurrency)

    yield  # -- server is running --

    log.info("Shutting down - releasing model backend.")
    _state["backend"] = None
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
# Exception Handler: unhandled server errors (structured fail-closed)
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    log.error(
        "LIVE_ERROR_EVENT | %s",
        json.dumps({
            "event_type": "unhandled_server_exception",
            "request_id": request_id,
            "exception_type": type(exc).__name__,
            "detail": str(exc),
            "path": request.url.path,
            "status_code": 500,
        }),
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
    )


# ---------------------------------------------------------------------------
# Middleware: request ID + rate limiting (with health/ready exemption)
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_request_id_and_rate_limit(request: Request, call_next):
    # Request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Exempt health and readiness probes from rate limiting
    path = request.url.path
    if path in ("/health", "/ready"):
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

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
        log.warning(
            "LIVE_ERROR_EVENT | %s",
            json.dumps({
                "event_type": "rate_limit_exceeded",
                "request_id": request_id,
                "client_ip": client_ip,
                "path": path,
                "requests_in_window": len(_rate_limit_store[client_ip]),
                "limit_per_minute": max_requests,
                "status_code": 429,
            }),
        )
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
    Run the active deepfake classifier on a 16kHz mono audio array.
    This runs SYNCHRONOUSLY and must be called from a thread pool.

    Returns:
        {
          "label": "bonafide" | "spoof",
          "confidence": float 0.0-1.0,
          "risk_level": "low" | "medium" | "high",
          "spoof_score": float 0.0-1.0,
          "prob_real": float 0.0-1.0,
          "model_backend": str,
          "score_type": str,
        }
    """
    backend_obj: Optional[ModelBackend] = _state.get("backend")
    if backend_obj is not None and backend_obj.is_loaded():
        pred = backend_obj.predict(audio, sample_rate=cfg.audio.sample_rate)
        prob_real = pred.prob_real
        prob_fake = pred.prob_fake
        backend_name = pred.backend_name
        score_type = pred.score_type
    elif _state.get("model") is not None and _state.get("feature_extractor") is not None:
        # Fallback for unit test fixtures mocking model/feature_extractor directly
        model = _state["model"]
        feature_extractor = _state["feature_extractor"]
        device = _state.get("device", "cpu")
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
        prob_real = float(probs[0].item())
        prob_fake = float(probs[1].item())
        backend_name = cfg.active_model
        score_type = cfg.active_backend.score_type
    else:
        raise RuntimeError("Model backend is not loaded.")

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
        "model_backend": backend_name,
        "score_type": score_type,
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


def _load_audio_file(file_path: str, target_sr: int) -> np.ndarray:
    """
    Robust audio loader supporting WAV, MP3, FLAC, OGG, M4A, AAC, and WEBM.
    Attempts librosa first; falls back to PyAV for formats without libsndfile/ffmpeg support.
    """
    try:
        audio, _ = librosa.load(file_path, sr=target_sr, mono=True)
        return audio
    except Exception as librosa_err:
        log.warning("librosa.load failed (%s) — falling back to PyAV for %s", librosa_err, file_path)
        try:
            import av
            container = av.open(file_path)
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
            frames = []
            for frame in container.decode(audio=0):
                for resampled_frame in resampler.resample(frame):
                    frames.append(resampled_frame.to_ndarray())
            if frames:
                return np.concatenate(frames, axis=1).squeeze(0)
            return np.array([], dtype=np.float32)
        except Exception as av_err:
            raise HTTPException(
                status_code=422,
                detail=f"Cannot decode audio file: librosa error: {librosa_err}; PyAV error: {av_err}",
            ) from av_err


def _make_insufficient_evidence_response(
    reason_codes: List[str],
    request_id: str,
    audio_quality: Optional[AudioQualityReport] = None,
    action: str = "verify",
    model_loaded: bool = True,
) -> DetectionResponse:
    """Create an INSUFFICIENT_EVIDENCE response."""
    aq = audio_quality.to_dict() if audio_quality else {}
    return DetectionResponse(
        decision="insufficient_evidence",
        action=action,
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
        model_backend=cfg.active_model,
        score_type=cfg.active_backend.score_type,
        threshold_version=cfg.thresholds.version,
        model_loaded=model_loaded,
        request_id=request_id,
    )


# Action mapping
_DECISION_ACTION_MAP = {
    "low": ("low_risk", "allow_with_caution"),
    "medium": ("verification_required", "verify"),
    "high": ("verification_required", "verify"),  # Single clip cannot generate action_held
}


def _check_live_window_complete(
    audio: np.ndarray,
    sample_rate: int,
    full_window_ms: int,
) -> bool:
    """
    Return True iff the decoded audio is at least full_window_ms long.

    This is the sole gate that prevents partial live windows from reaching
    model inference.  It is intentionally NOT applied to /predict — uploaded
    files may be any length that passes the normal quality checks.
    """
    duration_ms = int(len(audio) / sample_rate * 1000)
    return duration_ms >= full_window_ms


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
        raise HTTPException(
            status_code=503,
            detail="Model backend is unavailable. Check readiness at /ready.",
        )

    if not _state["model_loaded"]:
        return JSONResponse(
            status_code=503,
            content=_make_insufficient_evidence_response(
                ["model_unavailable"],
                request_id,
                action="unavailable",
                model_loaded=False,
            ).model_dump(),
        )

    # --- validate file type ---
    filename = file.filename or ""
    ext = os.path.splitext(filename)[-1].lower()
    if ext and ext not in cfg.audio.allowed_extensions:
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

        # Load audio (robust: librosa + PyAV fallback)
        audio = _load_audio_file(tmp_path, cfg.audio.sample_rate)

        if audio is None or len(audio) == 0:
            return _make_insufficient_evidence_response(
                ["empty_or_invalid_capture"],
                request_id,
                action="unavailable",
                model_loaded=_state["model_loaded"],
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
                action="verify",
                model_loaded=_state["model_loaded"],
            )

        # --- Run inference off the event loop ---
        sem = _inference_semaphore or asyncio.Semaphore(cfg.server.maximum_concurrency)
        async with sem:
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
        model_backend=result.get("model_backend", cfg.active_model),
        score_type=result.get("score_type", cfg.active_backend.score_type),
        threshold_version=cfg.thresholds.version,
        model_loaded=_state["model_loaded"],
        request_id=request_id,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness check — is the process alive?"""
    return HealthResponse(status="ok", model_loaded=_state["model_loaded"])


@app.get("/ready", response_model=ReadinessResponse)
async def ready():
    """Readiness check — is the model loaded and ready for inference?"""
    backend_obj = _state.get("backend")
    backend_name = backend_obj.backend_name if backend_obj else cfg.active_model
    score_type = backend_obj.score_type if backend_obj else cfg.active_backend.score_type

    if _state["model_loaded"]:
        return ReadinessResponse(
            status="ready",
            model_loaded=True,
            model_backend=backend_name,
            score_type=score_type,
            model_version=cfg.model.version,
            threshold_version=cfg.thresholds.version,
            device=_state.get("device", "unknown"),
            demo_mode=cfg.security.demo_mode,
        )
    return JSONResponse(
        status_code=503,
        content={
            "status": "not_ready",
            "model_loaded": False,
            "model_backend": backend_name,
            "score_type": score_type,
            "model_version": cfg.model.version,
            "threshold_version": cfg.thresholds.version,
            "device": "",
            "demo_mode": cfg.security.demo_mode,
            "detail": _state.get("load_error") or "Model backend is not loaded or failed initialization.",
        },
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

    Live-window contract (from cfg.live_analysis):
        full_window_ms = 4000 ms  — minimum for inference
        stride_ms      = 2000 ms  — rolling-buffer advance

    Any chunk shorter than full_window_ms is rejected with
    reason_codes=["partial_window"] and never reaches the model.
    This guard does NOT apply to /predict (upload path).
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    if not _state["model_loaded"]:
        return JSONResponse(
            status_code=503,
            content=_make_insufficient_evidence_response(
                ["model_unavailable"],
                request_id,
                action="unavailable",
                model_loaded=False,
            ).model_dump(),
        )

    # --- read with size limit ---
    try:
        contents = await _read_upload_limited(file, cfg.server.maximum_upload_bytes)
    except HTTPException:
        raise

    tmp_path: Optional[str] = None
    try:
        tmp_path = _save_upload_to_temp(contents, ".wav")

        # Load audio (robust: librosa + PyAV fallback)
        audio = _load_audio_file(tmp_path, cfg.audio.sample_rate)

        if audio is None or len(audio) == 0:
            return _make_insufficient_evidence_response(
                ["empty_or_invalid_capture"],
                request_id,
                action="unavailable",
                model_loaded=_state["model_loaded"],
            )

        # --- Complete-window guard (live path only) ---
        # A chunk shorter than FULL_WINDOW_MS must NOT reach inference.
        # This is the primary fix: partial startup buffers, timer-flush
        # fragments, and sub-4-second chunks are rejected here.
        # /predict is deliberately excluded from this guard.
        window_complete = _check_live_window_complete(
            audio, cfg.audio.sample_rate, cfg.live_analysis.full_window_ms
        )
        chunk_duration_ms = round(len(audio) / cfg.audio.sample_rate * 1000, 1)

        if not window_complete:
            log.info(
                "LIVE | request_id=%s | PARTIAL_WINDOW rejected | duration_ms=%.1f | required_ms=%d",
                request_id, chunk_duration_ms, cfg.live_analysis.full_window_ms,
            )
            return _make_insufficient_evidence_response(
                ["partial_window"],
                request_id,
                action="verify",
                model_loaded=_state["model_loaded"],
            )

        # --- Audio quality check (FULL_WINDOW_MS is the effective min_duration) ---
        quality = analyze_audio_quality(
            audio, cfg.audio.sample_rate,
            min_duration=cfg.live_analysis.full_window_seconds,
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
                action="verify",
                model_loaded=_state["model_loaded"],
            )

        # --- Run inference off the event loop ---
        sem = _inference_semaphore or asyncio.Semaphore(cfg.server.maximum_concurrency)
        async with sem:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                _inference_executor,
                _run_inference_sync,
                audio,
            )

    except HTTPException as http_exc:
        log.warning(
            "LIVE_ERROR_EVENT | %s",
            json.dumps({
                "event_type": "live_analyze_http_error",
                "request_id": request_id,
                "status_code": http_exc.status_code,
                "detail": str(http_exc.detail),
            }),
        )
        raise
    except Exception as exc:
        log.error(
            "LIVE_ERROR_EVENT | %s",
            json.dumps({
                "event_type": "live_analyze_processing_exception",
                "request_id": request_id,
                "exception_type": type(exc).__name__,
                "detail": str(exc),
            }),
            exc_info=True,
        )
        raise HTTPException(
            status_code=422,
            detail=f"Failed to process audio chunk ({type(exc).__name__}): {exc}",
        ) from exc
    finally:
        if tmp_path:
            _cleanup_temp(tmp_path)

    # --- Build clip-level response ---
    # Single clip cannot generate ACTION_HELD — that is a call-level decision
    decision, action = _DECISION_ACTION_MAP.get(
        result["risk_level"], ("verification_required", "verify")
    )

    log.info("LIVE | request_id=%s | label=%s | spoof_score=%.4f | risk=%s | duration_ms=%.1f",
             request_id, result["label"], result["spoof_score"], result["risk_level"], chunk_duration_ms)

    # Dev-mode sanitized event logging (never logs raw audio or PII)
    if cfg.security.demo_mode or os.environ.get("VOICEGUARD_DEV_MODE", "").strip() == "1" or not os.environ.get("VOICEGUARD_PROD"):
        log.info("LIVE_EVENT | %s", json.dumps({
            "request_id": request_id,
            "chunk_id": str(uuid.uuid4())[:8],
            "chunk_duration_ms": chunk_duration_ms,
            "full_window_ms": cfg.live_analysis.full_window_ms,
            "window_complete": window_complete,
            "sample_rate": cfg.audio.sample_rate,
            "channels": cfg.live_analysis.channels,
            "bits_per_sample": cfg.live_analysis.bits_per_sample,
            "bytes": len(contents),
            "decision": decision,
            "action": action,
            "label": result["label"],
            "spoof_score": result["spoof_score"],
            "score_type": result.get("score_type", cfg.active_backend.score_type),
            "speech_detected": True,
            "audio_quality_status": quality.status,
            "rms": round(quality.rms, 6),
            "voiced_ratio": round(quality.voiced_ratio, 4),
            "reason_codes": (["elevated_spoof_score"] if result["label"] == "spoof" else []),
            "model_backend": result.get("model_backend", cfg.active_model),
            "model_loaded": _state["model_loaded"],
            "response_age_ms": 0,
        }))

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
        model_backend=result.get("model_backend", cfg.active_model),
        score_type=result.get("score_type", cfg.active_backend.score_type),
        threshold_version=cfg.thresholds.version,
        model_loaded=_state["model_loaded"],
        request_id=request_id,
    )


# ---------------------------------------------------------------------------
# File Sliding Window Analysis endpoint
# ---------------------------------------------------------------------------
@app.post("/file/analyze_windows", response_model=FileSlidingAnalysisResponse)
async def analyze_file_windows(request: Request, file: UploadFile = File(...)):
    """
    Accept an uploaded audio file and analyze it using 4-second sliding windows
    with 2-second stride (kLiveFullWindowMs / kLiveStrideMs).

    Returns per-window acoustic and detection telemetry along with aggregated
    call-level metrics (EMA smoothing, persistence rule, action held).
    """
    if not _state["model_loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Model backend is unavailable. Check readiness at /ready.",
        )

    backend_obj = _state.get("backend")
    backend_name = backend_obj.backend_name if backend_obj else cfg.active_model
    score_type = backend_obj.score_type if backend_obj else cfg.active_backend.score_type

    filename = file.filename or ""
    ext = os.path.splitext(filename)[-1].lower()
    if ext and ext not in cfg.audio.allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(cfg.audio.allowed_extensions))}",
        )

    try:
        contents = await _read_upload_limited(file, cfg.server.maximum_upload_bytes)
    except HTTPException:
        raise

    suffix = ext if ext else ".wav"
    tmp_path: Optional[str] = None
    try:
        tmp_path = _save_upload_to_temp(contents, suffix)
        audio = _load_audio_file(tmp_path, cfg.audio.sample_rate)

        if audio is None or len(audio) == 0:
            return FileSlidingAnalysisResponse(
                total_duration_seconds=0.0,
                window_length_seconds=cfg.live_analysis.full_window_seconds,
                stride_seconds=cfg.live_analysis.stride_seconds,
                window_count=0,
                complete_window_count=0,
                high_risk_windows=0,
                maximum_spoof_score=None,
                average_spoof_score=None,
                ema_score=None,
                decision="insufficient_evidence",
                action="unavailable",
                reason_codes=["empty_or_invalid_capture"],
                model_backend=backend_name,
                model_version=cfg.model.version,
                score_type=score_type,
                threshold_version=cfg.thresholds.version,
                model_loaded=_state["model_loaded"],
                windows=[],
            )

        sr = cfg.audio.sample_rate
        total_duration = len(audio) / sr
        window_sec = cfg.live_analysis.full_window_seconds  # 4.0
        stride_sec = cfg.live_analysis.stride_seconds      # 2.0
        window_samples = int(window_sec * sr)             # 64000

        # Section 3: Short-file policy (< 4 seconds)
        if total_duration < window_sec:
            short_window = SlidingWindowItemResponse(
                window_index=0,
                start_seconds=0.0,
                end_seconds=round(total_duration, 2),
                duration_seconds=round(total_duration, 2),
                window_complete=False,
                decision="insufficient_evidence",
                action="unavailable",
                label="unknown",
                spoof_score=None,
                confidence=None,
                score_type=score_type,
                risk_level="unknown",
                speech_detected=False,
                quality_status="short_audio",
                rms=None,
                snr_db=None,
                voiced_ratio=None,
                reason_codes=["short_audio"],
                model_backend=backend_name,
                model_version=cfg.model.version,
                threshold_version=cfg.thresholds.version,
                model_loaded=_state["model_loaded"],
            )
            return FileSlidingAnalysisResponse(
                total_duration_seconds=round(total_duration, 2),
                window_length_seconds=window_sec,
                stride_seconds=stride_sec,
                window_count=1,
                complete_window_count=0,
                high_risk_windows=0,
                maximum_spoof_score=None,
                average_spoof_score=None,
                ema_score=None,
                decision="insufficient_evidence",
                action="unavailable",
                persistence_triggered=False,
                consecutive_high_count=0,
                reason_codes=["short_audio"],
                model_backend=backend_name,
                model_version=cfg.model.version,
                score_type=score_type,
                threshold_version=cfg.thresholds.version,
                model_loaded=_state["model_loaded"],
                windows=[short_window],
            )

        # Section 4: Complete 4-second windows with 2-second stride
        start_sec = 0.0
        window_idx = 0
        windows: List[SlidingWindowItemResponse] = []
        valid_scores: List[float] = []
        consecutive_high = 0
        persistence_triggered = False
        ema_score: Optional[float] = None

        sem = _inference_semaphore or asyncio.Semaphore(cfg.server.maximum_concurrency)
        loop = asyncio.get_running_loop()

        while start_sec + window_sec <= total_duration:
            start_sample = int(round(start_sec * sr))
            end_sample = start_sample + window_samples
            chunk = audio[start_sample:end_sample]

            quality = analyze_audio_quality(
                chunk, sr,
                min_duration=window_sec,
                max_duration=cfg.audio.maximum_duration_seconds,
                rms_silence_threshold=cfg.audio_quality.rms_silence_threshold,
                min_voiced_ratio=cfg.audio_quality.minimum_voiced_ratio,
                max_clipping_ratio=cfg.audio_quality.maximum_clipping_ratio,
                poor_snr_db=cfg.audio_quality.poor_snr_threshold_db,
            )

            if not quality.is_usable():
                w_reasons = quality.reason_codes or [f"audio_quality_{quality.status}"]
                windows.append(SlidingWindowItemResponse(
                    window_index=window_idx,
                    start_seconds=round(start_sec, 2),
                    end_seconds=round(start_sec + window_sec, 2),
                    duration_seconds=round(window_sec, 2),
                    window_complete=True,
                    decision="insufficient_evidence",
                    action="verify",
                    label="unknown",
                    spoof_score=None,
                    confidence=None,
                    score_type=score_type,
                    risk_level="unknown",
                    speech_detected=quality.status != "silent",
                    quality_status=quality.status,
                    rms=round(quality.rms, 6),
                    snr_db=round(quality.snr_db, 2) if quality.snr_db is not None else None,
                    voiced_ratio=round(quality.voiced_ratio, 4),
                    reason_codes=w_reasons,
                    model_backend=backend_name,
                    model_version=cfg.model.version,
                    threshold_version=cfg.thresholds.version,
                    model_loaded=_state["model_loaded"],
                ))
                consecutive_high = 0
            else:
                async with sem:
                    inf = await loop.run_in_executor(
                        _inference_executor,
                        _run_inference_sync,
                        chunk,
                    )

                spoof_score = inf["spoof_score"]
                confidence = inf["confidence"]
                risk_level = inf["risk_level"]
                label = inf["label"]

                valid_scores.append(spoof_score)

                alpha = cfg.risk_aggregation.ema_alpha
                if ema_score is None:
                    ema_score = spoof_score
                else:
                    ema_score = round(alpha * spoof_score + (1.0 - alpha) * ema_score, 4)

                if spoof_score >= cfg.thresholds.high_risk_threshold:
                    consecutive_high += 1
                else:
                    consecutive_high = 0

                if consecutive_high >= cfg.risk_aggregation.persistent_high_windows:
                    persistence_triggered = True

                w_decision, w_action = _DECISION_ACTION_MAP.get(
                    risk_level, ("verification_required", "verify")
                )
                w_reasons = ["elevated_spoof_score"] if label == "spoof" else []

                windows.append(SlidingWindowItemResponse(
                    window_index=window_idx,
                    start_seconds=round(start_sec, 2),
                    end_seconds=round(start_sec + window_sec, 2),
                    duration_seconds=round(window_sec, 2),
                    window_complete=True,
                    decision=w_decision,
                    action=w_action,
                    label=label,
                    spoof_score=spoof_score,
                    confidence=confidence,
                    score_type=score_type,
                    risk_level=risk_level,
                    speech_detected=True,
                    quality_status=quality.status,
                    rms=round(quality.rms, 6),
                    snr_db=round(quality.snr_db, 2) if quality.snr_db is not None else None,
                    voiced_ratio=round(quality.voiced_ratio, 4),
                    reason_codes=w_reasons,
                    model_backend=backend_name,
                    model_version=cfg.model.version,
                    threshold_version=cfg.thresholds.version,
                    model_loaded=_state["model_loaded"],
                ))

            window_idx += 1
            start_sec += stride_sec

        # Check for final partial segment (only if audio extends beyond the last complete window)
        last_end = windows[-1].end_seconds if windows else 0.0
        if last_end < round(total_duration, 2) - 0.05 and start_sec < total_duration:
            partial_dur = round(total_duration - start_sec, 2)
            windows.append(SlidingWindowItemResponse(
                window_index=window_idx,
                start_seconds=round(start_sec, 2),
                end_seconds=round(total_duration, 2),
                duration_seconds=partial_dur,
                window_complete=False,
                decision="insufficient_evidence",
                action="verify",
                label="unknown",
                spoof_score=None,
                confidence=None,
                score_type=score_type,
                risk_level="unknown",
                speech_detected=False,
                quality_status="partial_window",
                rms=None,
                snr_db=None,
                voiced_ratio=None,
                reason_codes=["partial_window"],
                model_backend=backend_name,
                model_version=cfg.model.version,
                threshold_version=cfg.thresholds.version,
                model_loaded=_state["model_loaded"],
            ))

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to process audio windows: {exc}",
        ) from exc
    finally:
        if tmp_path:
            _cleanup_temp(tmp_path)

    complete_count = sum(1 for w in windows if w.window_complete)
    high_risk_windows = sum(1 for w in windows if w.risk_level == "high" and w.window_complete)
    max_spoof = max(valid_scores) if valid_scores else None
    avg_spoof = round(float(np.mean(valid_scores)), 4) if valid_scores else None

    file_reasons: List[str] = []
    if not valid_scores:
        decision = "insufficient_evidence"
        action = "unavailable"
        file_reasons.append("no_valid_windows")
    elif persistence_triggered:
        decision = "action_held"
        action = "hold"
        file_reasons.append("persistent_high_risk")
    elif high_risk_windows > 0 or (ema_score is not None and ema_score >= cfg.risk_aggregation.high_ema_threshold):
        decision = "verification_required"
        action = "verify"
        file_reasons.append("elevated_spoof_score")
    elif any(w.risk_level == "medium" for w in windows if w.window_complete) or (ema_score is not None and ema_score >= cfg.risk_aggregation.medium_ema_threshold):
        decision = "verification_required"
        action = "verify"
    else:
        decision = "low_risk"
        action = "allow_with_caution"

    return FileSlidingAnalysisResponse(
        total_duration_seconds=round(total_duration, 2),
        window_length_seconds=window_sec,
        stride_seconds=stride_sec,
        window_count=len(windows),
        complete_window_count=complete_count,
        high_risk_windows=high_risk_windows,
        maximum_spoof_score=max_spoof,
        average_spoof_score=avg_spoof,
        ema_score=ema_score,
        decision=decision,
        action=action,
        persistence_triggered=persistence_triggered,
        consecutive_high_count=consecutive_high,
        reason_codes=file_reasons,
        model_backend=backend_name,
        model_version=cfg.model.version,
        score_type=score_type,
        threshold_version=cfg.thresholds.version,
        model_loaded=_state["model_loaded"],
        windows=windows,
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
