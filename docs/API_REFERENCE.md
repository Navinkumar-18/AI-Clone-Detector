# VoiceGuard API Reference

**Base URLs**:
- Demo Mode: `https://localhost:8443` (TLS self-signed) or `http://localhost:8000` (Plain HTTP dev mode)
- API Version: `2.0.0`
- Model: `garystafford/wav2vec2-deepfake-voice-detector` (revision `voiceguard-v1`)

---

## 1. Zero-Persistence Guarantee

VoiceGuard does **not** persist uploaded audio, raw PCM arrays, or voice biometrics to disk, database, or logs.
Temporary audio files created for format conversion are unlinked immediately in request `finally` blocks, followed by an explicit `PRIVACY` audit log entry. Only scalar prediction telemetry and audio quality metrics are returned to the caller.

---

## 2. Endpoints Overview

| Endpoint | Method | Purpose | Response Model |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | Process liveness check | `HealthResponse` |
| `/ready` | `GET` | Model & config readiness check | `ReadinessResponse` |
| `/predict` | `POST` | Full-file deepfake risk detection | `DetectionResponse` |
| `/live/analyze` | `POST` | Live chunk-level deepfake risk evaluation | `DetectionResponse` |

---

## 3. Endpoint Specifications

### `GET /health`
Liveness probe to confirm that the server process is responsive.

**Response `200 OK`**:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### `GET /ready`
Readiness probe to confirm that the PyTorch/HuggingFace model is materialized in memory and ready for inference.

**Response `200 OK`**:
```json
{
  "status": "ready",
  "model_loaded": true,
  "model_version": "voiceguard-v1",
  "threshold_version": "threshold-v2",
  "device": "cuda",
  "demo_mode": false
}
```

---

### `POST /predict`
Uploads an audio file (`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`) to evaluate synthetic speech indicators.

**Headers**:
- `Content-Type: multipart/form-data`

**Request Body**:
- `file`: Audio file binary (maximum size: 10,485,760 bytes / 10MB)

**Successful Response `200 OK`**:
```json
{
  "decision": "low_risk",
  "action": "allow_with_caution",
  "label": "bonafide",
  "spoof_score": 0.0412,
  "confidence": 0.9588,
  "risk_level": "low",
  "risk_percentage": 95.9,
  "speech_detected": true,
  "audio_quality": {
    "status": "acceptable",
    "duration_seconds": 3.42,
    "rms": 0.124,
    "snr_db": 22.4,
    "clipping_ratio": 0.0,
    "voiced_ratio": 0.78
  },
  "evidence": {
    "window_count": 0,
    "high_risk_window_count": 0,
    "analysis_age_ms": 0
  },
  "reason_codes": [],
  "model_version": "voiceguard-v1",
  "threshold_version": "threshold-v2",
  "request_id": "8f3b23c9-943f-4e09-b697-3f338d1502be"
}
```

**Quality Gated Response (e.g., Silent Audio) `200 OK`**:
```json
{
  "decision": "insufficient_evidence",
  "action": "verify",
  "label": "unknown",
  "spoof_score": 0.0,
  "confidence": 0.0,
  "risk_level": "unknown",
  "risk_percentage": 0.0,
  "speech_detected": false,
  "audio_quality": {
    "status": "silent",
    "duration_seconds": 2.0,
    "rms": 0.0,
    "snr_db": null,
    "clipping_ratio": 0.0,
    "voiced_ratio": 0.0
  },
  "evidence": {
    "window_count": 0,
    "high_risk_window_count": 0,
    "analysis_age_ms": 0
  },
  "reason_codes": [
    "no_speech"
  ],
  "model_version": "voiceguard-v1",
  "threshold_version": "threshold-v2",
  "request_id": "a9010f3c-589e-4e89-a228-3e4b77f901cb"
}
```

---

### `POST /live/analyze`
Uploads a short streaming chunk (typically 0.5s - 2.0s) for real-time call monitoring.

> **Important**: This endpoint outputs **clip-level** evidence (`verification_required` or `low_risk`). The final call-level status (`action_held`) is aggregated on the client side across multiple consecutive sliding windows.

**Response `200 OK`**: Matches `DetectionResponse` schema with window-level evidence.

---

## 4. Error Status Codes

- `413 Payload Too Large`: Upload exceeds `cfg.server.maximum_upload_bytes` (10 MB).
- `415 Unsupported Media Type`: Uploaded extension not in `allowed_extensions` (`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`).
- `422 Unprocessable Entity`: Audio cannot be parsed by audio decoding engines.
- `429 Too Many Requests`: Client exceeded `cfg.server.rate_limit_requests_per_minute` (60 req/min default).
