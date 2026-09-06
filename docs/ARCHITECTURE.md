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
│  │ - Model: garystafford/wav2vec2-deepfake-voice-detector                │  │
│  │ - Softmax scoring: prob_fake and prob_real                            │  │
│  └───────────────────────────────────┬───────────────────────────────────┘  │
│                                      │                                      │
│  ┌───────────────────────────────────▼───────────────────────────────────┐  │
│  │ Response Dispatch: Canonical DetectionResponse (Pydantic)             │  │
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
