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
│  └───────────────────────────────────┬───────────────────────────────────┘  │
│                                      │                                      │
│  ┌───────────────────────────────────▼───────────────────────────────────┐  │
│  │ Streaming Chunk Ingestion: 64 KB chunks, early 10 MB limit (HTTP 413) │  │
│  └───────────────────────────────────┬───────────────────────────────────┘  │
│                                      │                                      │
│  ┌───────────────────────────────────▼───────────────────────────────────┐  │
│  │ Ephemeral Transient Storage: /tmp/vg_<UUID>.wav                       │  │
│  │ Unconditional deletion in finally block with PRIVACY audit log        │  │
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
│  │ Decisions: low_risk | verification_required | action_held | uncertain │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Non-Blocking Inference & Concurrency Design

To prevent heavy CPU/GPU PyTorch operations from freezing the asynchronous FastAPI event loop, VoiceGuard implements an off-loop thread pool architecture:

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
   - Default concurrency limit: 4 parallel workers.
   - Prevents server thread exhaustion and out-of-memory (OOM) crashes under concurrent load.
3. **Event Loop Responsiveness**:
   - Liveness (`GET /health`) and readiness (`GET /ready`) probes execute immediately without waiting behind queued inference jobs.

---

## 3. Decision Matrix & Risk State Machine

VoiceGuard enforces four canonical states:

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
                         (Score Spike: Spoof ≥ 0.40)
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
1. **Silence Invariant**: Silence **never** transitions to `LOW_RISK` or `ALLOW_WITH_CAUTION`.
2. **Hold Preservation**: If the state is `ACTION_HELD` and silence or network disruption occurs, the state transitions to `INSUFFICIENT_EVIDENCE` while the action **remains held**.
3. **No Single-Spike Freezes**: An isolated high-risk chunk generates `VERIFICATION_REQUIRED`. Only persistent risk ($N=2$) triggers `ACTION_HELD`.
4. **Hysteresis Cooldown**: Escaping an active hold requires $M=3$ consecutive clean windows.
