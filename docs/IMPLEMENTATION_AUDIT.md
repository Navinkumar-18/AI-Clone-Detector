# VoiceGuard — Implementation Audit

**Base commit**: `45b9547abcb22c77bf7e6a1ebe972b96d5b3886d`  
**Target branch**: `voiceguard/sih2026-hardening`  
**Verification**: Verified on branch `voiceguard/sih2026-hardening`.  

---

## 1. Implemented Features

The following features are implemented, functional in the code, and validated by automated tests:

| Feature | Primary Location | Verification Method |
| :--- | :--- | :--- |
| **FastAPI Backend Server** | `backend.py` | Verified via pytest in `tests/test_backend.py` |
| **wav2vec2 Classifier** | `backend.py` | Uses `garystafford/wav2vec2-deepfake-voice-detector` (app version `voiceguard-v1`, upstream revision `c66306024a7ede0be291e9c4558b37634782dc4e`) |
| **Non-Blocking Inference Engine** | `backend.py` | Offloaded to `ThreadPoolExecutor` bounded by async semaphore |
| **Pre-Inference Audio Quality Filter** | `audio_quality.py` | RMS silence, clipping ratio, voiced frame ratio, and SNR gating in `tests/test_audio_quality.py` |
| **Silence Safety Invariant** | `audio_quality.py`, `backend.py` | Silence returns `insufficient_evidence` with `speech_detected: false`; never returns `bonafide` |
| **Sliding-Window Risk Aggregator** | `risk_aggregator.py` | Exponential Moving Average ($\alpha=0.3$), persistence ($N=2$), cooldown ($M=3$) in `tests/test_risk_aggregator.py` |
| **Canonical Response Schema** | `backend.py`, `lib/models/live_analysis_result.dart` | Pydantic model (`DetectionResponse`) synchronized with Dart in `tests/test_api_schema.py` |
| **Streaming Upload Byte Limits** | `backend.py` | Chunked 64 KB ingestion with early 10 MB limit (`HTTP 413`) |
| **In-Memory Rate Limiting** | `backend.py` | In-memory sliding window rate limiter (60 req/min, `HTTP 429`); single-process prototype only |
| **Readiness & Liveness Probes** | `backend.py` | `GET /health` (liveness) and `GET /ready` (model/config readiness) |
| **Scoped Demo TLS in Flutter** | `lib/api_client.dart`, `lib/config.dart` | Removed global HttpOverrides; `IOClient` restricts self-signed exceptions strictly to VoiceGuard host when `kDemoMode && !kReleaseMode` |
| **Single Source Configuration** | `config/model_config.yaml`, `voiceguard_config.py` | YAML configuration with cached loader and environment variable overrides |
| **Deterministic Test Fixtures** | `tests/conftest.py` | 8 synthetic audio fixtures covering edge cases |
| **Automated Verification Suite** | `tests/` | 54 unit and integration tests passing in pytest |

---

## 2. Partially Implemented Features

| Feature | Current State | Remaining Limitation / Future Work |
| :--- | :--- | :--- |
| **Distributed Rate Limiting** | In-memory per-process rate limiting | In multi-worker production deployments, an external Redis or memory-grid backend is required. |
| **Client-Side Live Audio Streaming** | Microphone audio is recorded in 4-second WAV chunks | Chunks are sent via HTTP POST rather than continuous bidirectional WebSockets. |
| **Acoustic Environment Normalization** | Baseline SNR and clipping checks | Full noise-cancellation (spectral subtraction / Wiener filtering) is not yet applied before feature extraction. |

---

## 3. Simulated Features

| Feature | Simulation Details | Why Simulated |
| :--- | :--- | :--- |
| **Sensitive Transaction Protection** | The mock transaction card shows a ₹2,00,000 transfer to "ABC Suppliers" with an action button that disables on `ACTION_HELD` or `INSUFFICIENT_EVIDENCE`. Disclaimer: "Demo mode — no real financial transaction is executed." | Real banking APIs (UPI, IMPS, NEFT) require commercial NPCI / banking licenses and regulatory sandbox approval. |
| **Out-of-Band Step-Up Challenge** | Clicking "Verify Caller First" displays simulated step-up verification options, such as confirming through the official app, calling a saved number, or contacting a trusted person. No real financial or telephony service is invoked. | Demonstrates the fraud-prevention workflow without invoking external telecom or banking gateways. |

---

## 4. Demo-Only Features

| Feature | Description | Production Requirement |
| :--- | :--- | :--- |
| **Local Self-Signed TLS Certificate** | Automatically generated via `generate_cert.py` if missing on localhost. | Production systems must use CA-signed certificates (e.g., Let's Encrypt or corporate PKI) with public trust chains. |
| **Opt-In Demo TLS Exception** | `kDemoMode` allows connecting to localhost HTTPS with self-signed certs. | Must remain disabled (`false`) in production builds. Scoped strictly to demo runs. |
| **Dev Mode HTTP Launch** | `VOICEGUARD_DEV_MODE=1` allows binding plain HTTP on port 8000. | Strictly prohibited on untrusted networks or production environments. |

---

## 5. Security Limitations

1. **Compromised Key Material in Git History**: `cert.pem` and `key.pem` were previously committed and must be treated as compromised. They have been untracked from the branch and ignored via `.gitignore`, but historical removal was not automatically performed.
2. **Audio Air-Gap Eavesdropping**: When using speakerphone mode, ambient room audio is captured by the microphone.
3. **No Caller Identity Proof**: The system evaluates synthetic generation characteristics. It cannot detect a human imposter.
4. **Ephemeral Storage Boundaries**: Operating-system memory, swap files, crash dumps, client buffers, infrastructure logs, and backups are outside the guarantees of this prototype.

---

## 6. Features That Cannot Be Verified Locally

1. **Real-World Cellular Network Codec Degradation**: Performance on live 2G GSM (AMR 4.75 kbps) calls cannot be verified in a local desktop development environment.
2. **Non-English / Multilingual Synthetic Speech**: Model was evaluated on English speech; Indian regional languages (Hindi, Tamil, Telugu, Marathi, etc.) cannot be verified without labelled regional corpora.

---

## 7. Features That Must NOT Be Claimed Publicly

| Claim to Avoid | Accurate Technical Positioning |
| :--- | :--- |
| ⛔ "VoiceGuard verifies speaker identity." | VoiceGuard detects deepfake acoustic artifacts; it does not authenticate caller identity. |
| ⛔ "VoiceGuard achieves 99% accuracy on voice clones." | Baseline pretrained models exhibit score overlap across benchmarks; the threshold report records high FPR on out-of-domain evaluation. |
| ⛔ "VoiceGuard blocks real UPI/bank transactions." | The transaction workflow is a simulated hackathon prototype. |
| ⛔ "VoiceGuard intercepts native phone calls on mobile OS." | Mobile OS sandboxes prohibit native call tapping; VoiceGuard uses speakerphone audio capture. |
| ⛔ "Softmax outputs are calibrated probabilities." | Scores are raw heuristic softmax outputs, not calibrated probabilities. |
| ⛔ "Zero-persistence privacy guarantee or DPDP compliance." | Ephemeral audio processing with cleanup logging; no formal legal/regulatory compliance assessment has been performed. |
