# VoiceGuard API Reference

**Base URLs**:
- Demo Mode: `https://localhost:8443` (TLS self-signed) or `http://localhost:8000` (Plain HTTP dev mode)
- API Version: `2.0.0`
- Model: `garystafford/wav2vec2-deepfake-voice-detector` (version `voiceguard-v1`, upstream revision `c66306024a7ede0be291e9c4558b37634782dc4e`)

---

## 1. Ephemeral Audio Processing

VoiceGuard is designed for ephemeral audio processing. Temporary audio is processed for inference and cleaned up after processing. Structured cleanup events are logged without intentionally storing raw audio content.

Temporary audio files created during upload handling are unlinked in request `finally` blocks, followed by a structured `PRIVACY` cleanup log entry. Only scalar prediction telemetry and audio quality metrics are returned to the caller.

> [!WARNING]
> Operating-system memory, swap files, crash dumps, client buffers, infrastructure logs, and backups are outside the guarantees of this prototype.

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
Readiness probe to confirm that the model is loaded in memory and ready for inference.

**Response `200 OK`**:
```json
{
  "status": "ready",
  "model_loaded": true,
  "model_version": "voiceguard-v1",
  "threshold_version": "threshold-v2",
  "device": "cpu",
  "demo_mode": false
}
```
*If model weights are still loading, returns HTTP 503 or `status: "not_ready"`.*

---

### `POST /predict`
Upload an audio file (.wav, .flac, .mp3, .ogg, .m4a, .aac, .webm) for comprehensive analysis.

**Form Data**:
- `file`: Audio file binary (maximum 10 MB).

**Response `200 OK`**:
```json
{
  "decision": "low_risk",
  "action": "allow_with_caution",
  "label": "bonafide",
  "spoof_score": 0.0412,
  "confidence": 0.9588,
  "risk_level": "low",
  "risk_percentage": 4.12,
  "speech_detected": true,
  "audio_quality": {
    "status": "clean",
    "rms_energy": 0.042,
    "snr_db": 22.4,
    "clipping_ratio": 0.0,
    "voiced_frame_ratio": 0.45,
    "duration_seconds": 3.8
  },
  "evidence": {
    "acoustic_anomaly": 0.0412,
    "spectral_flatness": 0.0,
    "pitch_stability": 0.0
  },
  "reason_codes": [],
  "model_version": "voiceguard-v1",
  "threshold_version": "threshold-v2",
  "request_id": "8f3b23c9-943f-4e09-b697-3f338d1502be"
}
```

---

### `POST /live/analyze`
Receives a sliding 4-second audio window for real-time risk assessment.

**Response `200 OK`**: Identical canonical `DetectionResponse` schema as `/predict`.
