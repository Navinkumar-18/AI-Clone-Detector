# VoiceGuard — System Architecture

**System Version**: 2.0.0  
**Target Solution**: Real-Time Voice-Clone Risk Detection & Step-Up Verification  

---

## 1. High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FLUTTER CLIENT APPLICATION                       │
│                                                                             │
│  ┌────────────────────────┐    ┌────────────────────────┐                   │
│  │ Audio Capture Pipeline │    │  Sliding Risk          │                   │
│  │ - 16kHz PCM Stream     │───►│  Aggregator Engine     │                   │
│  │ - Rolling WAV Buffer   │    │  - EMA Filter (α=0.3)  │                   │
│  └────────────────────────┘    │  - Persistence (N=2)   │                   │
│                                │  - Cooldown (M=3)      │                   │
│                                └───────────┬────────────┘                   │
│                                            │                                │
│  ┌────────────────────────┐                ▼                                │
│  │ UI & Step-Up Challenge │    ┌────────────────────────┐                   │
│  │ - Risk Gauge / Waves   │◄───│  Simulated Transaction │                   │
│  │ - Step-Up Out-of-Band  │    │  Card (Held / Caution) │                   │
│  └────────────────────────┘    └────────────────────────┘                   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────┐                   │
│  │ Scoped IOClient (Demo TLS exception restricted only  │                   │
│  │ to configured VoiceGuard host when kDemoMode=true)   │                   │
│  └──────────────────────────┬───────────────────────────┘                   │
└─────────────────────────────┼───────────────────────────────────────────────┘
                              │ HTTPS / TLS 1.3
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FASTAPI BACKEND ENGINE                             │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Middleware: Request ID (X-Request-ID) + IP Rate Limiter (60 req/min)  │  │
│  │ (In-memory rate limiter is for single-process demo only)              │  │
│  └───────────────────────────────────┬───────────────────────────────────┘  │
│                                      │                                      │
│  ┌───────────────────────────────────▼───────────────────────────────────┐  │
│  │ Streaming Chunk Ingestion: 64 KB chunks, early 10 MB limit (HTTP 413) │  │
│  └───────────────────────────────────┬───────────────────────────────────┘  │
│                                      │                                      │
│  ┌───────────────────────────────────▼───────────────────────────────────┐  │
│  │ Ephemeral Transient Storage: /tmp/vg_<UUID>.wav                       │  │
│  │ Cleanup in finally block with structured PRIVACY cleanup log          │  │
│  └───────────────────────────────────┬───────────────────────────────────┘  │
│                                      │                                      │
│  ┌───────────────────────────────────▼───────────────────────────────────┐  │
│  │ Pre-Inference Audio Quality Gate (audio_quality.py):                  │  │
│  │ - RMS silence check  - Clipping ratio  - Voiced frame ratio  - SNR    │  │
│  │ Unusable audio immediately returns INSUFFICIENT_EVIDENCE             │  │
│  └───────────────────────────────────┬───────────────────────────────────┘  │
│                                      │ Acceptable Quality                   │
│  ┌───────────────────────────────────▼───────────────────────────────────┐  │
│  │ Non-Blocking Inference Worker Pool:                                   │  │
│  │ - Bounded ThreadPoolExecutor (max_workers=4)                          │  │
│  │ - Guarded by asyncio.Semaphore(4) concurrency ceiling                 │  │
│  │ - Model Backend Factory (voiceguard.model_backends):                 │  │
│  │   * Wav2Vec2Backend (default): uncalibrated_softmax_score             │  │
│  │   * WavLMMLPBackend (candidate): uncalibrated_sigmoid_score           │  │
│  └───────────────────────────────────┬───────────────────────────────────┘  │
│                                      │                                      │
│  ┌───────────────────────────────────▼───────────────────────────────────┐  │
│  │ Response Dispatch: Canonical DetectionResponse (Pydantic)             │  │
│  │ Fields: spoof_score, confidence, model_backend, score_type            │  │
│  │ Decisions: low_risk | verification_required | action_held             │  │
│  │            | insufficient_evidence                                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Non-Blocking Inference & Concurrency Design

To prevent synchronous PyTorch calculations from blocking the asynchronous FastAPI event loop:

1. **Thread Pool Offload**:
   ```python
   loop = asyncio.get_running_loop()
   result = await loop.run_in_executor(
       _inference_executor,
       _run_inference_sync,
       audio,
   )
   ```
2. **Concurrency Limiting**:
   - Governed by `_inference_semaphore = asyncio.Semaphore(cfg.server.maximum_concurrency)`.
   - Bounded concurrency limit: 4 parallel workers.
3. **Event Loop Responsiveness**:
   - Liveness (`GET /health`) and readiness (`GET /ready`) probes execute immediately without waiting behind queued inference tasks.

### 2.1 Pluggable Model-Backend Abstraction Layer

VoiceGuard decouples the HTTP serving lifecycle and audio quality gates from the underlying neural architecture via `voiceguard/model_backends/`:

- **`ModelBackend` Base Interface** (`voiceguard/model_backends/base.py`):
  Defines standard lifecycle methods (`load()`, `predict(audio, sample_rate) -> ModelPrediction`) and attributes (`backend_name`, `score_type`).
- **`Wav2Vec2Backend`** (`voiceguard/model_backends/wav2vec2_backend.py`):
  Production default. Wraps Hugging Face `AutoModelForAudioClassification` (`garystafford/wav2vec2-deepfake-voice-detector` at pinned revision `c66306024a7ede0be291e9c4558b37634782dc4e`). Emits `score_type: "uncalibrated_softmax_score"`.
- **`WavLMMLPBackend`** (`voiceguard/model_backends/wavlm_mlp_backend.py`):
  Offline research candidate. Extracts frozen 768-dimensional representations via `microsoft/wavlm-base` (pinned revision `efa81aae7ff777e464159e0f877d54eac5b84f81`) and scores them with a 2-layer MLP head (`models/best_mlp_wavlm_base.pt`). Emits `score_type: "uncalibrated_sigmoid_score"`.
- **Backend Factory** (`voiceguard/model_backends/__init__.py`):
  Dynamically instantiates the configured backend via `create_backend(backend_name, ...)`. If a backend fails to load, raises `ModelLoadError`, ensuring the server fails closed without implicit fallback.


---

## 3. Decision Matrix & Risk State Machine

```
                          ┌────────────────────────┐
                          │  INSUFFICIENT_EVIDENCE │◄─── (Silence / Poor SNR /
                          │  Action: Verify / Hold │      Backend Unreachable)
                          └───────────┬────────────┘
                                      │
                               (Speech Detected)
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │        LOW_RISK        │
                          │ Action: Allow w/Caution│
                          └───────────┬────────────┘
                                      │
                         (Score Spike: Spoof ≥ 0.30)
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │  VERIFICATION_REQUIRED │
                          │ Action: Verify Caller  │
                          └───────────┬────────────┘
                                      │
                         (Persistent: 2 Chunks ≥ 0.85)
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │       ACTION_HELD      │
                          │ Action: Hold Transact. │
                          └───────────┬────────────┘
                                      │
                         (Cooldown: 3 Clean Chunks < 0.40)
                                      │
                                      ▼
                          (Return to VERIFICATION_REQUIRED
                                  → LOW_RISK)
```

### Safety Invariants:
1. **Silence Invariant**: Digital silence **never** returns `bonafide` or `allow_with_caution`.
2. **Hold Preservation**:
   ```text
   ACTION_HELD
       → silence/backend failure
       → INSUFFICIENT_EVIDENCE
       → sensitive action remains held
   ```
3. **No Automatic Allowance**: A sensitive action never automatically becomes allowed because of silence, timeout, stale data, or backend failure.
4. **Hysteresis Cooldown**: Escaping an active hold requires 3 consecutive clean windows.
