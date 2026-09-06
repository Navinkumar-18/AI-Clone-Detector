# VoiceGuard — Real-Time Voice-Clone & Audio-Deepfake Risk Detection

> **Smart India Hackathon (SIH 2026) Prototype Solution**  
> *Privacy-aware, real-time risk assessment and step-up verification for AI voice-clone impersonation attacks in financial and emergency transactions.*

---

## 📌 Executive Summary

With the advent of commercial generative speech models and one-shot voice cloning, malicious actors can clone a trusted person's voice from a 3-second audio snippet to bypass verbal authentication, deceive call center operators, and execute fraudulent money transfers.

**VoiceGuard** is an end-to-end security system combining deep learning acoustic artifact detection, audio quality gating, multi-window risk aggregation, and step-up challenge verification to mitigate voice cloning fraud without storing caller voiceprints.

```
 [Live Audio Stream / Audio File]
               │
               ▼
   [Audio Quality Pre-Filter] ────► Digital silence / low SNR ──► [INSUFFICIENT_EVIDENCE]
               │ (Acceptable Quality)
               ▼
   [wav2vec2 Feature Extractor] ──► Ephemeral inference off event loop
               │
               ▼
    [Acoustic Deepfake Model]  ──► Softmax Spoof Score & Confidence
               │
               ▼
  [Client-Side Risk Aggregator] ──► Multi-window temporal persistence & cooldown
               │
               ▼
   [Action / Decision Matrix]  ──► LOW_RISK / VERIFICATION_REQUIRED / ACTION_HELD
```

---

## 🛡️ Core Pillars & Architecture

### 1. Zero-Persistence Privacy Architecture
- **No Voice Biometrics**: VoiceGuard does not enroll, map, or store voice templates or biometric identifiers.
- **Ephemeral Processing**: Uploaded audio is processed strictly within a transient request scope and immediately purged from disk.
- **Audit-Logged Deletion**: An explicit `PRIVACY` log verifies file unlinking at the conclusion of every request.

### 2. Audio Quality & Insufficient-Evidence Gating
Traditional ML classifiers often produce confident but meaningless predictions when fed silence, white noise, or truncated buffers. VoiceGuard implements strict pre-inference signal validation in `audio_quality.py`:
- **Silence & Low-Energy Detection**: RMS energy below threshold immediately returns `insufficient_evidence` with `no_speech` reason code.
- **Clipping & Distortion Monitoring**: Rejects saturated waveforms that degrade acoustic features.
- **Speech Voicing Check**: Confirms harmonic speech presence before running heavy transformers.
- **Safety Guarantee**: Digital silence **never** yields a "bonafide" result and **never** clears an active security hold.

### 3. Client-Side Risk Aggregator & Fraud Protection
- **No Single-Spike False Triggers**: A single anomaly or audio glitch cannot freeze a high-value transaction.
- **Persistent Confirmation**: Requires $N$ consecutive high-risk sliding windows (default $N=2$) before elevating to `ACTION_HELD`.
- **Step-Up Verification**: Rather than abrupt call termination, VoiceGuard triggers secondary out-of-band verification (SMS OTP, security questions, in-app challenge).
- **Hysteresis Cooldown**: Restoring trust requires sustained clean windows ($M=3$), preventing attack oscillations.

### 4. Enterprise Security Defaults
- **Scoped TLS Bypass**: Flutter bypasses self-signed certificates **only** in local demo mode (`kDemoMode`), guarded by compile-time assertions for release builds.
- **DoS Safeguards**: Enforces strict upload limits (10 MB maximum) via streaming chunk verification, concurrency limits via async semaphores, and rate-limiting per client IP.
- **Audited Secrets**: `.gitignore` strictly blocks all `.pem`, `.crt`, and `.key` artifacts.

---

## 📊 Model Evaluation & Honest Technical Disclosure

VoiceGuard utilizes the `garystafford/wav2vec2-deepfake-voice-detector` pre-trained architecture.

> [!IMPORTANT]
> **Evaluation Status**:  
> The committed threshold report (`threshold_report.txt`) records a 100% false-positive rate at the reviewed threshold (0.40). A complete threshold sweep must be regenerated before making any claim about all thresholds.  
> The available evaluation artifact shows substantial score overlap between bona-fide and spoof samples, indicating poor class separation under this evaluation.

### Responsible Claims Guidance
- **What VoiceGuard Does**: Measures acoustic deepfake characteristics and triggers step-up verification when synthetic speech indicators are detected.
- **What VoiceGuard Does Not Claim**: Does not verify caller identity, does not claim calibrated probability distributions, and does not claim flawless accuracy across out-of-domain telephony or Indian regional languages without domain fine-tuning.

For comprehensive analysis, see [`docs/MODEL_EVALUATION.md`](file:///c:/Users/akina/sih2026/docs/MODEL_EVALUATION.md) and [`docs/KNOWN_LIMITATIONS.md`](file:///c:/Users/akina/sih2026/docs/KNOWN_LIMITATIONS.md).

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10+
- Flutter 3.x with Dart 3.x
- Git

### 1. Install Dependencies
```bash
# Python Backend Dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Flutter Client Dependencies
flutter pub get
```

### 2. Start the Backend Server
```bash
python backend.py
```
*The server automatically generates a self-signed TLS cert if needed and binds to `https://0.0.0.0:8443`.*  
*(For development without HTTPS: `VOICEGUARD_DEV_MODE=1 python backend.py` on port `8000`).*

Verify that the backend is ready:
```bash
curl -k https://localhost:8443/ready
```

### 3. Launch the Flutter Application
```bash
# Run on Windows desktop
flutter run -d windows

# Or run in Chrome / Web
flutter run -d chrome
```

---

## 🧪 Verification & Test Suite

The project includes a comprehensive 54-test automated Python test suite and zero-warning Flutter static analysis:

```bash
# Run all unit and integration tests
python -m pytest tests/ -v

# Run Flutter / Dart static analysis
dart analyze
```

### Test Coverage Highlights:
- `tests/test_backend.py`: Endpoints (`/health`, `/ready`, `/predict`, `/live/analyze`), rate limiting, upload limits, request tracing, mock inference.
- `tests/test_audio_quality.py`: Digital silence, clipping, SNR, duration gates, `insufficient_evidence` behavior.
- `tests/test_risk_aggregator.py`: Persistence windows, cooldown recovery, silence resilience, backend error states.
- `tests/test_api_schema.py`: Canonical Pydantic schema validation, config synchronization, Flutter cross-compatibility.
- `tests/test_security_config.py`: Secure defaults, demo mode guards, CORS restrictions.

---

## 📖 Documentation Index

- [Architecture & Data Flow](file:///c:/Users/akina/sih2026/docs/ARCHITECTURE.md)
- [Hackathon Demo Walkthrough & Judge Q&A](file:///c:/Users/akina/sih2026/docs/DEMO_GUIDE.md)
- [API Reference Specification](file:///c:/Users/akina/sih2026/docs/API_REFERENCE.md)
- [Model Evaluation & Claims](file:///c:/Users/akina/sih2026/docs/MODEL_EVALUATION.md)
- [Known Limitations & Roadmap](file:///c:/Users/akina/sih2026/docs/KNOWN_LIMITATIONS.md)
- [Security Remediation Audit](file:///c:/Users/akina/sih2026/docs/SECURITY_REMEDIATION.md)
- [Implementation Audit Trail](file:///c:/Users/akina/sih2026/docs/IMPLEMENTATION_AUDIT.md)

---

## ⚖️ License & Ethical Usage
This software prototype is built for educational, defensive, and fraud-mitigation research in accordance with Smart India Hackathon guidelines.
