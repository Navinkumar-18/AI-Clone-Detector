# VoiceGuard SIH 2026 — Hardening Walkthrough Report

**Repository**: `Navinkumar-18/AI-Clone-Detector`  
**Base Commit**: `45b9547abcb22c77bf7e6a1ebe972b96d5b3886d`  
**Target Branch**: `voiceguard/sih2026-hardening`  
**Evaluation Status**: Verified against actual branch, test outputs, and evaluation artifacts  

---

## 1. Repository and Branch Identity

The VoiceGuard hardening implementation and verification were executed on branch `voiceguard/sih2026-hardening` based on commit `45b9547abcb22c77bf7e6a1ebe972b96d5b3886d`. All file paths referenced throughout this documentation are repository-relative paths. Local machine paths (e.g. `file:///...`) are strictly excluded.

**Final Product Positioning**:  
> "VoiceGuard is a privacy-aware prototype for detecting suspicious synthetic-speech characteristics and requesting independent verification before a sensitive action. It processes audio ephemerally, checks audio quality before inference, aggregates evidence across multiple windows, and never treats silence or unavailable analysis as proof that a caller is genuine. The current model has significant evaluation limitations, so the prototype deliberately uses risk-based step-up verification rather than claiming identity authentication or real transaction blocking."

---

## 2. Baseline Audit

An initial technical audit of base commit `45b9547` identified critical security, machine learning, and architectural gaps:

1. **Tracked Certificate and Private Key**: `cert.pem` and `key.pem` were tracked in the Git working tree.
2. **Global TLS Bypass in Flutter**: A global `HttpOverrides` implementation in `lib/main.dart` disabled certificate validation across the entire Dart process.
3. **Flawed Silence Logic**: Silent audio clips (RMS < 0.003) returned `{"label": "bonafide", "confidence": 0.9999}`, erroneously treating digital silence as genuine speech.
4. **Denial-of-Service Exposure**: Uploads were read into memory using unvalidated `await file.read()`, lacking streaming size limits, concurrency controls, or rate limits.
5. **Overstated Evaluation Claims**: Benchmark documentation lacked class-separated distributions, concealing a 100% false-positive rate on genuine clips at threshold 0.30 on the evaluated subset.
6. **Simulated Workflows Lacked Disclaimers**: UI transaction cards did not prominently identify financial workflows as simulated demonstrations.

The detailed catalog of initial audit findings is recorded in `docs/IMPLEMENTATION_AUDIT.md`.

---

## 3. Secret Remediation & Certificate-Key Hygiene

1. **Git Tracking and Exclusion**:
   - Removed `cert.pem` and `key.pem` from Git tracking on the hardening branch using `git rm --cached cert.pem key.pem`.
   - Added certificate and key patterns (`*.pem`, `*.key`, `*.crt`, `*.pfx`, `*.p12`) to `.gitignore`.
   - Verified that no tracked certificate or private key files exist on the branch:
     ```bash
     git ls-files | grep -E '(^|/)(cert|key)\.(pem|crt|key)$' || true
     # Output: empty (zero tracked keys)
     ```
2. **Key Compromise Disclosure**:
   - Removed `cert.pem` and `key.pem` from Git tracking on the hardening branch and added certificate/key patterns to `.gitignore`. Because these files were previously committed, the old private key must be treated as compromised. Historical removal was not automatically performed.
   - `cert.pem` and `key.pem` were previously committed and must not be reused for public deployment. New certificates and keys must be generated. Git-history cleanup may be required separately.
   - `.gitignore` prevents future tracking of new keys; it does not retroactively rewrite Git history.
3. **Ephemeral Development Certificates**:
   - `generate_cert.py` generates local self-signed certificates dynamically on startup if absent. These local certificates are for development-only testing. Public deployment requires new trusted certificate and key material issued by a recognized Certificate Authority.

---

## 4. Unified Configuration & Response Schema

1. **Single Source of Truth**:
   - Created `config/model_config.yaml` as the authoritative configuration for all thresholds, audio parameters, server limits, and risk aggregation settings.
   - Threshold constants are centralized: `spoof_threshold: 0.30`, `high_risk_threshold: 0.85`. Independent threshold definitions in application code have been eliminated.
2. **Model Identity Separation**:
   - Model source: `garystafford/wav2vec2-deepfake-voice-detector`
   - Application model version: `voiceguard-v1`
   - Exact upstream model revision: `c66306024a7ede0be291e9c4558b37634782dc4e` (upstream Hugging Face commit hash)
   - Note: `voiceguard-v1` is the application wrapper version, distinct from the upstream model commit hash.
3. **Typed Configuration Loader**:
   - Implemented `voiceguard_config.py` with immutable dataclasses, cached loading, and environment variable override support (`VOICEGUARD_DEMO_MODE`, `VOICEGUARD_SPOOF_THRESHOLD`).
4. **Canonical Response Schema**:
   - Standardized FastAPI Pydantic schema `DetectionResponse` in `backend.py` with fields: `decision`, `action`, `label`, `spoof_score`, `confidence`, `risk_level`, `risk_percentage`, `speech_detected`, `audio_quality`, `evidence`, `reason_codes`, `model_version`, `threshold_version`, and `request_id`.
   - Synchronized Dart client models (`LiveAnalysisResult` and `PredictResult`). Validated by `tests/test_api_schema.py`.

---

## 5. Audio-Quality Gating & Insufficient-Evidence Safety

1. **Signal Quality Engine (`audio_quality.py`)**:
   - Evaluates RMS energy, clipping ratio, voiced frame ratio (zero-crossing rate heuristic), and Signal-to-Noise Ratio (SNR) before inference.
   - Gating criteria: minimum duration (0.5s), maximum duration (60.0s), RMS silence threshold (0.003), minimum voiced ratio (0.15), maximum clipping ratio (0.05), and poor SNR threshold (5.0 dB).
2. **Safety Invariants**:
   - Unusable audio immediately returns `decision="insufficient_evidence"`, `speech_detected=false`, and descriptive `reason_codes`.
   - Digital silence **never** returns `bonafide` or `allow_with_caution`.
   - Silence or backend failure after `ACTION_HELD` produces `INSUFFICIENT_EVIDENCE`.
   - Silence or backend failure does not clear an existing security hold.
   - A sensitive action never automatically becomes allowed because of silence, timeout, stale data, or backend failure.
   - Explicit state transition:
     ```text
     ACTION_HELD
         → silence/backend failure
         → INSUFFICIENT_EVIDENCE
         → sensitive action remains held
     ```
   - Validated by 11 unit tests in `tests/test_audio_quality.py`.

---

## 6. Temporal Risk Aggregation State Machine

1. **Multi-Window State Machine (`risk_aggregator.py`)**:
   - Implements Exponential Moving Average (EMA, $\alpha = 0.3$) smoothing across sliding audio windows.
   - Exponential Moving Average smoothing reduces the impact of isolated score spikes; persistent high evidence is required before an action hold.
   - Does not imply that smoothing eliminates all false positives.
2. **Precise Transition Rules**:
   - A single high-risk window produces `VERIFICATION_REQUIRED`.
   - Two consecutive configured high-risk windows ($\ge 0.85$) produce `ACTION_HELD`.
   - Three configured clean windows ($< 0.40$) are required for cooldown recovery.
   - Silence, network timeout, or backend failure transitions to `INSUFFICIENT_EVIDENCE` and **does not** clear an active `ACTION_HELD` state.
3. **Verification**:
   - Validated by 15 tests in `tests/test_risk_aggregator.py`.

---

## 7. Backend Hardening & Ephemeral Audio Handling

1. **Ephemeral Audio Handling**:
   - Implemented ephemeral audio handling: temporary audio is processed for inference and cleaned up after processing. Structured cleanup events are logged without storing raw audio content.
   - Temporary files (`/tmp/vg_<UUID>.wav`) are unlinked in the request `finally` block via `_cleanup_temp()`.
   - Structured privacy log emitted: `PRIVACY | Temp audio purged — no persistent storage of raw audio`.
   - **Qualification**: Operating-system memory, swap files, crash dumps, client buffers, infrastructure logs, and backups are outside the guarantees of this prototype.
   - Does not claim zero-persistence enforcement, guaranteed deletion, or cryptographic audit logging.
   - Designed around data-minimization and ephemeral-processing principles. This prototype has not undergone a formal legal or regulatory compliance assessment.
2. **DoS Defense & Resource Bounds**:
   - Streaming upload reader `_read_upload_limited()` processes 64 KB chunks, rejecting uploads exceeding 10 MB with `HTTP 413`.
   - Bounded `ThreadPoolExecutor` (4 workers) and `asyncio.Semaphore(4)` isolate synchronous PyTorch inference from the async event loop.
   - In-memory sliding-window IP rate limiter enforces 60 requests/minute (`HTTP 429`). Documented limitation: this in-memory rate limiter is suitable for a single-process prototype only and is not sufficient for distributed production deployment.
3. **Health & Readiness**:
   - `GET /health` provides lightweight liveness verification.
   - `GET /ready` verifies model materialization, device allocation, and threshold metadata.
4. **Verification**:
   - Validated by 12 tests in `tests/test_backend.py`.

---

## 8. Flutter Integration & Scoped Client TLS

1. **Scoped TLS Client Architecture**:
   - Removed the global `HttpOverrides` bypass. Restricted the development-only self-signed certificate exception to the `VoiceGuardApiClient`, scoped to the configured target host and enabled only when `kDemoMode && !kReleaseMode`.
   - Secure mode is the default (`kDemoMode` defaults to `false` in `lib/config.dart`).
   - Release builds cannot enable the demo bypass (`!kReleaseMode`).
   - Local self-signed certificates are for development-only testing. Public deployment requires new trusted certificate/key material.
   - The behavior is strictly a client-scoped exception, not a "scoped global TLS bypass."
2. **UI Telemetry & Honest Labels**:
   - Replaced misleading labels (`VOICE VERIFIED`, `Authentic Speaker`) with honest operational telemetry: `LOW DETECTED SPOOF EVIDENCE`, `Continue with caution`, `VERIFICATION REQUIRED`, `ACTION HELD`, and `INSUFFICIENT EVIDENCE`.
   - Displays audio quality indicators (energy, SNR, clipping, voiced ratio) and reason codes directly in the interface.
3. **Simulated Transaction Demonstration**:
   - `TransactionCard` (`lib/widgets/transaction_card.dart`) displays mock ₹2,00,000 transfer with prominent disclaimer:
     `"Demo mode — no real financial transaction is executed."`
   - Action button automatically locks on `ACTION_HELD` or `INSUFFICIENT_EVIDENCE`.
4. **End-to-End Scope**:
   - Local prototype execution passed: backend readiness, API communication, Flutter integration, and mock transaction flow were verified.
   - Native cellular call interception, real telephony integration, UPI integration, bank integration, and real transaction blocking were not tested and are outside the prototype scope.
   - Local prototype verification must not be used as evidence of production readiness.
5. **Static Analysis & Tests**:
   - `dart analyze`: No issues found.
   - `flutter analyze`: No issues found.
   - `flutter test`: 4 client tests passing.

---

## 9. Model Evaluation Scope, Limitations & Diagnosis

1. **Evaluation Scope**:
   - Evaluated a 250-clip subset of the ASVspoof 2019 LA evaluation set: 50 bona-fide and 200 spoof samples.
   - This subset result must not be interpreted as full-dataset performance or real-world call performance.
   - The evaluation script uses the same model and preprocessing path configured for the live backend.
2. **Empirical Results at Threshold 0.30**:
   - Evaluated clips: 250 (50 genuine, 200 spoof)
   - TP = 155, TN = 0, FP = 50, FN = 45
   - False Positive Rate (FPR): **1.0000 (100%)**
   - False Negative Rate (FNR): **0.2250 (22.5%)**
   - Precision: 0.7561, Recall: 0.7750, F1 Score: 0.7654
3. **Scientific Evaluation Statement**:
   > "On the evaluated subset of 50 bona-fide and 200 spoof clips, using the production model and threshold 0.30, all 50 bona-fide samples were classified above the spoof threshold, producing an observed FPR of 100% on this subset. The committed threshold report records a 100% false-positive rate at the reviewed threshold. A complete threshold sweep is required before making any claim about all thresholds."
4. **Diagnostic Findings on the 100% FPR**:
   - **Softmax Index Mapping**: Confirmed that `model.config.id2label` maps index 0 to `'real'` and index 1 to `'fake'`. In `backend.py`, `prob_real = probs[0].item()` and `prob_fake = probs[1].item()`. The index mapping matches correctly and is not inverted.
   - **Threshold 0.30 Origin**: Selected during initial sweeps of `evaluate_live_model.py` against `wav2vec2-deepfake-voice-detector` by maximizing the aggregate F1 score (76.54%) on the imbalanced 250-clip subset (200 spoof vs 50 bonafide). Because 155 true positives masked the 50 false positives in harmonic mean calculations, F1 was maximized at threshold 0.30 despite zero true negatives (TN=0). In contrast, the research WavLM pipeline (`predict.py`) used an offline MLP architecture with bonafide-probability output.
   - **Preprocessing Parity**: The preprocessing in `backend.py` (`librosa.load(..., sr=16000, mono=True)` + `feature_extractor(..., sampling_rate=16000, padding=True)`) functionally matches `evaluate_live_model.py` and `evaluation/run_production_evaluation.py`. The high FPR is not caused by a preprocessing discrepancy.
   - **Raw Score Distribution**:
     | Class | N | Min | p25 | Median | p75 | Max | Mean |
     | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
     | **bonafide** | 50 | **0.8107** | 0.8699 | **0.8976** | 0.9877 | **0.9952** | **0.9080** |
     | **spoof** | 200 | **0.0316** | 0.3756 | **0.8620** | 0.8932 | **0.9953** | **0.7050** |
   - **Scientific Assessment**: The score overlap is consistent with poor class separation under this evaluation, although the exact cause requires further investigation. Because every bona-fide clip in this subset received `prob_fake >= 0.8107` (with mean bona-fide score 0.9080 exceeding spoof mean 0.7050), bona-fide scores are not merely miscalibrated; the representations exhibit severe out-of-domain distribution shift.

---

## 10. Documentation Files

| File | Purpose | Verification Status |
| :--- | :--- | :--- |
| `README.md` | Executive overview, architecture, safe claims, and setup | Verified |
| `docs/HARDENING_WALKTHROUGH.md` | Comprehensive technical walkthrough and audit report | Verified |
| `docs/IMPLEMENTATION_AUDIT.md` | Feature breakdown: implemented, partial, simulated, and demo-only | Verified |
| `docs/SECURITY_REMEDIATION.md` | Key compromise disclosure, scoped client TLS, and DoS controls | Verified |
| `docs/ARCHITECTURE.md` | System data flow, non-blocking threading, and state machines | Verified |
| `docs/MODEL_EVALUATION.md` | Empirical benchmark results, FPR disclosure, and score distributions | Verified |
| `docs/KNOWN_LIMITATIONS.md` | Operational boundaries, ML limitations, and privacy boundaries | Verified |
| `docs/API_REFERENCE.md` | Canonical endpoint contracts and response models | Verified |
| `docs/DEMO_GUIDE.md` | Setup steps, troubleshooting, demo flows, and 5-minute judge script | Verified |
| `docs/THREAT_MODEL.md` | Threat vectors, mitigations, and prototype trust boundaries | Verified |
| `docs/DATA_FLOW_AND_PRIVACY.md` | Ephemeral audio lifecycle and data minimization principles | Verified |
| `docs/DEMO_DATASET.md` | ASVspoof protocol subset structure and synthetic test fixtures | Verified |

---

## 11. Test Evidence

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```
- **Result**: 54 passed in 7.95s (100% pass rate).
- **Test Scope**: 54 automated tests passed in the available test suite. These tests validate implementation behavior, API contracts, audio-quality gates, backend behavior, security configuration, and temporal risk aggregation. They do not establish real-world voice-deepfake detection accuracy.
- **Coverage Summary**:
  ```text
  Overall repository coverage: 39%
  ```
  Security-critical modules have targeted high coverage, while overall repository coverage is lower because the report includes broader application and supporting code.
  - `audio_quality.py`: 95%
  - `backend.py`: 79%
  - `voiceguard_config.py`: 90%
  - `risk_aggregator.py`: 82%
- **Static Analysis & Flutter Verification**:
  - `python -m compileall -q .`: Clean compilation across all files (exit code 0).
  - `python -c "from backend import app; print('OK')"`: Exits 0, prints `OK`.
  - Flutter and Dart static analysis completed without reported diagnostics in the recorded environment (`dart analyze` and `flutter analyze` reported 0 issues).
  - `flutter test`: 4 tests passed.

---

## 12. Demo Commands & Quick Start

### Complete Setup Instructions:

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

# 3. Verify configuration integrity
python -c "from voiceguard_config import get_config; print(get_config())"

# 4. Start backend server
python backend.py

# 5. Verify local readiness probe (in separate terminal)
curl -k https://localhost:8443/ready

# 6. Run automated test suites
pytest tests/ -v
flutter test
dart analyze

# 7. Launch Flutter application in demo mode
flutter run -d windows --dart-define=DEMO_MODE=true
```

### Operational Guidance:
- **Local Certificate Generation**: If `cert.pem` or `key.pem` is absent, `generate_cert.py` automatically generates a local self-signed certificate for `localhost:8443`.
- **Model Download**: On first start, Hugging Face downloads model weights (~360 MB) to the local cache. Subsequent launches load immediately from disk.
- **Readiness Handling (HTTP 503)**: If `GET /ready` returns HTTP 503 or `{"status": "not_ready"}`, model weights are still loading; wait 15–30 seconds for initialization.
- **Demo Mode Toggle**: In Flutter, passing `--dart-define=DEMO_MODE=true` enables local self-signed certificate handling scoped to the VoiceGuard host. In standard mode (`DEMO_MODE=false`), standard strict TLS validation is enforced.
- **Security Scope**: `curl -k` is used only for local self-signed testing. Public deployment must not use `curl -k`. The Flutter demo flag enables local client certificate handling only. The backend remains in secure mode. Flutter demo mode is enabled only to permit local self-signed certificate testing. This is not a public-deployment configuration.

### Demonstration Phrasing:
- Play a prepared synthetic or converted-speech sample through the supported microphone/speakerphone demonstration path.
- VoiceGuard presents simulated step-up verification options, such as confirming through the official app, calling a saved number, or contacting a trusted person. No real financial or telephony service is invoked.
- Prominent disclaimer: *"Demo mode — no real financial transaction is executed."*

---

## 13. Known Limitations

1. **Acoustic Domain Shift**: The model exhibits score overlap on the ASVspoof 2019 LA evaluation subset, producing a 100% observed false positive rate at threshold 0.30.
2. **Dataset & Language Coverage**: Evaluated on a 250-clip English subset. Regional Indian languages, regional accents, and noisy environments require domain fine-tuning and evaluation.
3. **Telephony Codec Degradation**: Performance over 8 kHz AMR/GSM cellular compression has not been established.
4. **In-Memory Rate Limiting**: The sliding-window rate limiter is single-process only; multi-instance deployment requires distributed state storage (e.g. Redis).
5. **Simulated Financial Protection**: The prototype demonstrates step-up verification flows. No native banking APIs, payment gateways, or telephony networks are integrated.
6. **Ephemeral Storage Boundaries**: Operating-system memory, swap files, crash dumps, client buffers, infrastructure logs, and backups are outside the guarantees of this prototype.

---

## 14. Remaining Blockers

- **Security & Implementation Blockers**: Zero critical blockers on branch `voiceguard/sih2026-hardening`. No tracked private keys exist, no insecure TLS bypass is active by default, and silence safely produces `INSUFFICIENT_EVIDENCE`.
- **Machine Learning Production Blockers**:
  1. Full evaluation sweep across all 73,566 ASVspoof 2019 LA clips and out-of-domain benchmarks.
  2. Investigation into representation overlap on bona-fide studio recordings.
  3. Probability calibration (e.g. temperature scaling or Platt scaling) before interpreting outputs as probabilities.
  4. Regional language and telephony codec fine-tuning.

---

## 15. Controlled-Demo Readiness Rating

```text
Controlled prototype demonstration readiness: 8/10
ML production readiness: 3/10
Real financial and native telephony deployment: not ready
```

The prototype is suitable for a controlled SIH demonstration after the claims and documentation are corrected. It is not ready for real financial, identity, or telephony deployment.
