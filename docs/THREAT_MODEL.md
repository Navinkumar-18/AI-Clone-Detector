# VoiceGuard — Threat Model

**Document Version**: 1.0  
**Target System**: VoiceGuard Real-Time Voice-Clone Risk Detection & Step-Up Verification  
**Audience**: Security Reviewers, SIH Evaluators, and Systems Engineers  

---

## 1. System Overview & Protected Assets

VoiceGuard protects users from financial fraud and unauthorized actions initiated via AI voice-cloning (deepfake speech) and social engineering impersonation attacks.

### Protected Assets:
1. **User Financial Assets**: Prevention of unauthorized fund transfers (e.g., mock ₹2,00,000 transaction).
2. **Caller Voice Privacy**: Protection of user audio streams from unauthorized storage, surveillance, or biometric harvesting.
3. **Backend Service Availability**: Protection of inference pipelines from denial-of-service (DoS) and resource exhaustion attacks.

---

## 2. Threat Actors & Capabilities

| Threat Actor | Motivation | Technical Capability | Typical Vector |
| :--- | :--- | :--- | :--- |
| **Cybercriminal Syndicate** | Financial theft via executive / family impersonation | High (Diffusion/Autoregressive TTS, VC models, RVC) | Injecting cloned speech during phone conversations requesting urgent wire transfers |
| **Opportunistic Fraudster** | Social engineering / vishing | Low to Medium (Replaying pre-recorded samples, soundboards) | Replay attacks, ambient background manipulation |
| **Malicious Network Adversary** | Traffic tampering, credential sniffing, DoS | Medium to High (MITM, traffic injection) | Attempting TLS downgrade, uploading gigabyte files to crash backend |

---

## 3. Trust Boundaries & Assumptions

```
 [ Untrusted Environment ]           [ Client Trust Boundary ]          [ Server Trust Boundary ]
 ┌──────────────────────┐            ┌───────────────────────────┐      ┌───────────────────────────┐
 │ Remote Caller Stream │ ──(Audio)─►│ Smartphone Microphone /   │      │ VoiceGuard FastAPI Server │
 │ (Human / Cloned AI)  │            │ Speakerphone Capture      │      │                           │
 └──────────────────────┘            │                           │      │ - Audio Quality Filter    │
                                     │ - Sliding Risk Aggregator │      │ - wav2vec2 Deepfake Model │
                                     │ - Transaction Interceptor │      │ - Ephemeral Temp Storage  │
                                     │ - Scoped TLS Client       │      └───────────────────────────┘
                                     └─────────────┬─────────────┘                    ▲
                                                   │                                  │
                                                   └───[ HTTPS / TLS 1.3 ]────────────┘
                                                       (Encrypted Transient Chunks)
```

### Core Security Assumptions:
1. **Zero-Trust Network**: All communications between client and server are assumed to traverse potentially hostile networks. Transport Layer Security (TLS) is mandatory.
2. **Untrusted Input Stream**: Audio input is untrusted and may contain silence, clipped noise, corrupted file headers, or adversarial payloads.
3. **Zero-Persistence Guarantee**: The server must never write caller voiceprints, raw PCM arrays, or audio files to persistent databases or long-term storage.

---

## 4. Attack Vectors & Implemented Mitigations

### Attack 1: Real-Time Generative Voice-Cloning Impersonation
- **Description**: Adversary trains or fine-tunes a voice-cloning model (e.g., ElevenLabs, XTTS, RVC) on 3 seconds of target voice to authorize a sensitive transfer.
- **Mitigation**: Pretrained wav2vec2 feature extractor detects acoustic generation artifacts (spectral discontinuities, phase inconsistencies, vocoder signatures). Softmax spoof scores elevate risk.
- **Defense-in-Depth**: Client-side **Risk Aggregator** requires 2 consecutive elevated windows before triggering `ACTION_HELD`, holding the transaction and forcing out-of-band step-up verification.

### Attack 2: Silence / Muted Microphone Evasion
- **Description**: Adversary stays silent or mutes microphone during sensitive transaction prompts to avoid detection, attempting to bypass security with false bonafide scores.
- **Mitigation**: Pre-inference **Audio Quality Engine** (`audio_quality.py`) measures RMS energy. Signals below silence threshold return `insufficient_evidence` with `speech_detected: false`.
- **Security Invariant**: Silence **never** returns `bonafide` and **never** clears an active `ACTION_HELD` state.

### Attack 3: Single-Spike Audio Glitch Manipulation
- **Description**: Transient network packets, cellular dropout, or coughing causes an isolated artifact, which in a naive system would permanently disconnect legitimate users.
- **Mitigation**: Exponential Moving Average ($\alpha = 0.3$) smoothing combined with persistence rules. A single spike generates `VERIFICATION_REQUIRED`, while only persistent evidence locks `ACTION_HELD`. Recovery requires 3 consecutive clean windows (hysteresis).

### Attack 4: Denial-of-Service via Oversized Payloads (Zip Bombs / Massive Audio)
- **Description**: Adversary streams multi-gigabyte files to crash server memory.
- **Mitigation**: `_read_upload_limited()` streams uploads in 64 KB chunks and enforces a strict 10 MB maximum limit, rejecting with `HTTP 413 Payload Too Large` before loading into RAM. Decoded audio duration is capped at 30 seconds.

### Attack 5: Concurrency Overload & Event Loop Starvation
- **Description**: Flooding concurrent inference requests blocks async I/O handlers.
- **Mitigation**: PyTorch inference is decoupled from the asyncio event loop into a bounded `ThreadPoolExecutor` governed by an `asyncio.Semaphore`. Excess traffic is throttled via per-client IP rate limiting (60 req/min, `HTTP 429`).

### Attack 6: Man-in-the-Middle (MITM) & Compromised Certificate Re-use
- **Description**: Using tracked repository certificates to intercept live audio traffic.
- **Mitigation**: Previously tracked certificates (`cert.pem`, `key.pem`) were purged from the git tree and declared compromised. `.gitignore` blocks all `.pem`, `.crt`, and `.key` extensions. Insecure TLS bypass is disabled by default (`kDemoMode = false`) and forbidden in production release builds. Demo TLS exceptions in `IOClient` are strictly scoped only to the configured VoiceGuard host.

---

## 5. Residual Risks & Out-of-Scope Threats

1. **Human Social Engineering Without Voice Cloning**: An authentic human fraudster tricking a victim is outside the scope of deepfake acoustic detection.
2. **Severe Telephony Compression (2G GSM / AMR 4.75kbps)**: Extreme low-bitrate telephony can strip high-frequency vocoder cues, increasing false positive rates.
3. **Compromised Operating System / Rooted Device**: If the host device OS is compromised at the kernel level, local microphone audio or display memory could be intercepted.
