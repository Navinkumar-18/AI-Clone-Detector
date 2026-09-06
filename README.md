# VoiceGuard — Real-Time Voice-Clone & Audio-Deepfake Risk Detection

> **Smart India Hackathon (SIH 2026) Solution Prototype**  
> "VoiceGuard is a privacy-aware prototype for detecting suspicious synthetic-speech characteristics and requesting independent verification before a sensitive action. It processes audio ephemerally, checks audio quality before inference, aggregates evidence across multiple windows, and never treats silence or unavailable analysis as proof that a caller is genuine. The current model has significant evaluation limitations, so the prototype deliberately uses risk-based step-up verification rather than claiming identity authentication or real transaction blocking."

---

## 📌 Executive Summary

With the proliferation of commercial generative speech synthesis and one-shot voice cloning, malicious actors can clone voices to deceive victims, bypass verbal verification, and attempt fraudulent money transfers.

**VoiceGuard** evaluates acoustic synthetic-speech characteristics using a wav2vec2 architecture, screens signal quality prior to inference, aggregates risk across sliding windows, and presents step-up verification challenges before sensitive actions proceed.

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

### 1. Ephemeral Audio Processing
- VoiceGuard is designed for ephemeral audio processing. Temporary audio is processed for inference and cleaned up after processing. Structured cleanup events are logged without intentionally storing raw audio content.
- **Privacy Qualification**: Operating-system memory, swap files, crash dumps, client buffers, infrastructure logs, and backups are outside the guarantees of this prototype.
- **Compliance Qualification**: Designed around data-minimization and ephemeral-processing principles. This prototype has not undergone a formal legal or regulatory compliance assessment.

### 2. Audio Quality & Insufficient-Evidence Gating
Traditional classifiers can produce arbitrary, high-confidence outputs on silent or truncated buffers. VoiceGuard enforces pre-inference signal validation in `audio_quality.py`:
- **Silence & Low-Energy Detection**: RMS energy below threshold immediately returns `insufficient_evidence` with `no_speech` reason code.
- **Clipping & Distortion Monitoring**: Rejects heavily clipped waveforms that distort acoustic spectral representations.
- **Speech Voicing Check**: Checks harmonic voiced frames before invoking neural feature extractors.
- **Safety Invariants**: Digital silence **never** yields a "bonafide" result, **never** produces `allow_with_caution`, and **never** clears an active security hold.
- **State Transition**:
  ```text
  ACTION_HELD
      → silence/backend failure
      → INSUFFICIENT_EVIDENCE
      → sensitive action remains held
  ```

### 3. Client-Side Risk Aggregation & Step-Up Verification
- **Isolated Spikes**: Exponential Moving Average smoothing reduces the impact of isolated score spikes; persistent high evidence is required before an action hold. (Does not imply that smoothing eliminates all false positives.)
- **Persistent Confirmation**: A single high-risk window produces `VERIFICATION_REQUIRED`. Two consecutive configured high-risk windows produce `ACTION_HELD`.
- **Step-Up Verification**: VoiceGuard presents simulated step-up verification options, such as confirming through the official app, calling a saved number, or contacting a trusted person. No real financial or telephony service is invoked.
- **Hysteresis Cooldown**: Restoring low risk requires three consecutive clean windows.
- **Simulated Transaction Disclaimer**: Demo mode — no real financial transaction is executed.

### 4. Security & Hardened Defaults
- **Scoped TLS Exception**: Removed the global `HttpOverrides` bypass. Restricted the development-only self-signed certificate exception to the `VoiceGuardApiClient`, scoped to the configured target host and enabled only when `kDemoMode && !kReleaseMode`. Secure mode is the default (`kDemoMode` defaults to `false`). Release builds cannot enable the demo bypass.
- **Certificate-Key Hygiene**: Removed `cert.pem` and `key.pem` from Git tracking on the hardening branch and added certificate/key patterns to `.gitignore`. Because these files were previously committed, the old private key must be treated as compromised. Historical removal was not automatically performed. Public deployment requires new trusted certificate/key material.
- **DoS Safeguards**: Enforces strict upload limits (10 MB ceiling) via streaming chunk verification, concurrency limits via async semaphores and thread pools, and rate-limiting per client IP. The in-memory rate limiter is suitable for a single-process prototype only and is not sufficient for distributed production deployment.

---

## 📊 Model Identity & Evaluation Disclosure

### Model Identification:
```text
Model source: garystafford/wav2vec2-deepfake-voice-detector
Application model version: voiceguard-v1
Exact upstream model revision: c66306024a7ede0be291e9c4558b37634782dc4e
```

### Evaluation Scope:
The evaluation script uses the same model and preprocessing path configured for the live backend.

> [!IMPORTANT]
> **Evaluation Scope & Exact Scientific Statement**:  
> Evaluated a 250-clip subset of the ASVspoof 2019 LA evaluation set: 50 bona-fide and 200 spoof samples. This subset result must not be interpreted as full-dataset performance or real-world call performance.  
> On the evaluated subset of 50 bona-fide and 200 spoof clips, using the production model and threshold 0.30, all 50 bona-fide samples were classified above the spoof threshold, producing an observed FPR of 100% on this subset. The committed threshold report records a 100% false-positive rate at the reviewed threshold. A complete threshold sweep is required before making any claim about all thresholds.  
> The score overlap is consistent with poor class separation under this evaluation, although the exact cause requires further investigation.

### Raw Score Distribution on Evaluated Subset:
| Class | N | Min | p25 | Median | p75 | Max | Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **bonafide** | 50 | **0.8107** | 0.8699 | **0.8976** | 0.9877 | **0.9952** | **0.9080** |
| **spoof** | 200 | **0.0316** | 0.3756 | **0.8620** | 0.8932 | **0.9953** | **0.7050** |

For comprehensive technical discussion, see `docs/MODEL_EVALUATION.md` and `docs/KNOWN_LIMITATIONS.md`.

---

## 🚀 Quick Start Guide

### Setup Commands:

```bash
# 1. Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\activate

# Linux/macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3. Verify configuration
python -c "from voiceguard_config import get_config; print(get_config())"

# 4. Start backend server
python backend.py

# 5. Check local readiness probe (in separate terminal)
curl -k https://localhost:8443/ready

# 6. Run automated test suites
pytest tests/ -v
flutter test
dart analyze

# 7. Launch Flutter application in demo mode
flutter run -d windows --dart-define=DEMO_MODE=true
```

### Operational Notes:
- **Self-Signed Certificates**: Generated dynamically by `generate_cert.py` for local loopback testing if absent. Local self-signed certificates are for development-only testing. Public deployment requires new trusted certificate/key material.
- **Model Download**: Model weights (~360 MB) download automatically on first run to the local Hugging Face cache.
- **Readiness Handling (HTTP 503)**: If `GET /ready` returns HTTP 503 or `{"status": "not_ready"}`, model weights are still loading into memory; wait 15–30 seconds.
- **Security Scope**: `curl -k` is used only for local self-signed testing; public deployment must not use `curl -k`. The backend remains in secure mode. Flutter demo mode is enabled only to permit local self-signed certificate testing. This is not a public-deployment configuration.

---

## 🧪 Verification & Test Suite

54 automated tests passed in the available test suite. These tests validate implementation behavior, API contracts, audio-quality gates, backend behavior, security configuration, and temporal risk aggregation. They do not establish real-world voice-deepfake detection accuracy.

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

```text
Overall repository coverage: 39%
```
Security-critical modules have targeted high coverage (`audio_quality.py`: 95%, `voiceguard_config.py`: 90%, `risk_aggregator.py`: 82%, `backend.py`: 79%), while overall repository coverage is lower because the report includes broader application and supporting code.

Flutter and Dart static analysis completed without reported diagnostics in the recorded environment (`dart analyze` and `flutter analyze` reported 0 issues).

Local prototype execution passed: backend readiness, API communication, Flutter integration, and mock transaction flow were verified. Native cellular call interception, real telephony integration, UPI integration, bank integration, and real transaction blocking were not tested and are outside the prototype scope.

---

## 🏷️ Controlled-Demo Readiness Rating

```text
Controlled prototype demonstration readiness: 8/10
ML production readiness: 3/10
Real financial and native telephony deployment: not ready
```

The prototype is suitable for a controlled SIH demonstration after the claims and documentation are corrected. It is not ready for real financial, identity, or telephony deployment.

---

## 📖 Documentation Index

- [Hardening Walkthrough & Audit Report](docs/HARDENING_WALKTHROUGH.md)
- [Architecture & Data Flow](docs/ARCHITECTURE.md)
- [Hackathon Demo Guide & Judge Script](docs/DEMO_GUIDE.md)
- [API Reference Specification](docs/API_REFERENCE.md)
- [Model Evaluation & Findings](docs/MODEL_EVALUATION.md)
- [Known Limitations](docs/KNOWN_LIMITATIONS.md)
- [Security Remediation Audit](docs/SECURITY_REMEDIATION.md)
- [Implementation Audit Trail](docs/IMPLEMENTATION_AUDIT.md)
- [Threat Model](docs/THREAT_MODEL.md)
- [Data Flow & Privacy](docs/DATA_FLOW_AND_PRIVACY.md)

---

## ⚖️ Ethical Usage Notice
This software prototype is built for educational, defensive, and fraud-mitigation research in accordance with Smart India Hackathon guidelines.
